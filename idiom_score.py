import torch as t 
from collections import Counter, defaultdict
import pandas as pd
import json
from tqdm import tqdm
from transformer_lens import (
    HookedTransformer,
    head_detector
)
from merge_tokenizers import PythonGreedyCoverageAligner, types
import os
import einops

class IdiomScorer:
    def __init__(self, model, filename: str = "pythia_formal_idiom_pos_idiom_score.json", start: int = 0, cage_dir: str = ""):
        """
        This class calculates the Idiom Score.

        @params
            model: examined model
            filename: file with the idiom positions
            start: sentence idx to start with (only needed for caging)
            cage_dir: path to the caged activations
        """
        self.model = model
        self.model.cfg.use_attn_result = True
        self.aligner = PythonGreedyCoverageAligner()
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.scores = None
        self.idiom_positions = self.load_all_idiom_pos(filename)
        self.components = None
        self.features = 5
        self.batch_num = start
        self.cage_dir = cage_dir

    def get_all_idiom_pos(self, batch):
        """
        This method extracts the idiom positions of a batch.

        @params
            batch: batch of sentences
        """
        for i in range(len(batch["sentence"])):
            sent = batch["sentence"][i]
            model_str_tokens= self.model.to_str_tokens(sent)
            aligned_positions = self.align_tokens(sent, batch["tokenized"][i], model_str_tokens)
            self.idiom_positions.append(self.get_idiom_pos(aligned_positions, batch["tags"][i]))

    def store_all_idiom_pos(self, filename: str):
        """
        This method saves the idiom positions.

        @params
          filename: output file
        """
        with open(filename, 'w', encoding = "utf-8") as f:
            json.dump(self.idiom_positions, f)    

    def load_all_idiom_pos(self, filename: str):
        """
        This methods loads the idiom positions.

        @params
            filename: file with the idiom positions
        @returns the loaded idiom positions
        """
        if os.path.isfile(filename):
            with open(filename, 'r', encoding = "utf-8") as f:
                return json.load(f) 
        else:
            return [] 

    def get_cache(self, sent):
        """
        This method creates the activation cache for a sentence.

        @params
            sent: processed sentence
        @returns
            activation cache on the device
        """
        idiom_tokens = self.model.to_tokens(sent)
        _, cache = self.model.run_with_cache(idiom_tokens, remove_batch_dim=True)
        return cache.to(self.device)

    def get_cage(self):
        """
        This method loads the caged results.

        @returns 
            pattern_stack: stack of attention patterns
            result_stack: stack of attention results
        """
        if os.path.isfile(f"{self.cage_dir}/pattern/{self.batch_num}.pt"):
            pattern_stack = t.load(f"{self.cage_dir}/pattern/{self.batch_num}.pt", map_location=t.device(self.device))
        else:
            pattern_stack = None
        
        if os.path.isfile(f"{self.cage_dir}/result/{self.batch_num}.pt"):
            result_stack = t.load(f"{self.cage_dir}/result/{self.batch_num}.pt", map_location=t.device(self.device))
            result_stack = t.einsum("ijkl->ikjl", result_stack)
        else:
            result_stack = None
        
        return pattern_stack, result_stack


    def create_idiom_features(self, sent: str, idiom_pos: tuple):
        """
        This method computes the components for the Idiom Score. 

        @params
            sent: processed sentence
            idiom_pos: start and end position of the idiom
        @returns
            layer_head_features: featurs for all layers and heads
        """
        pattern_stack, result_stack = self.get_cage()
        cache = None
        if pattern_stack == None or result_stack == None:
            cache = self.get_cache(sent)

        layer_head_features = t.zeros(self.model.cfg.n_layers, self.model.cfg.n_heads, self.features, dtype=t.float16, device = self.device) 
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                if cache != None:
                    attention_pattern = cache["pattern", layer][head].to(dtype=t.float16)
                    head_result = cache["result", layer][:, head, :].to(dtype=t.float16) # seq x heads x d_model
                else:
                    attention_pattern = pattern_stack[layer][head]
                    head_result = result_stack[layer][head]
                layer_head_features[layer][head] = self.compute_components(attention_pattern, head_result, idiom_pos)

                del attention_pattern
                del head_result
                t.cuda.empty_cache()
        
        del cache
        del pattern_stack
        del result_stack
        t.cuda.empty_cache()

        return layer_head_features
    
    def create_idiom_score_tensor(self, batch, comp_file: str):
        """
        This method computes the idiom score for a batch of sentences.

        @params
            batch: batch of instances
            comp_file: file for the intermediate feature tensor results
        """
        batch_scores = t.zeros(len(batch["sentence"]), self.model.cfg.n_layers, self.model.cfg.n_heads, self.features, dtype=t.float16, device = self.device)

        for i in range(len(batch["sentence"])):
            batch_scores[i] = self.create_idiom_features(batch["sentence"][i], batch["idiom_pos"][i])
        
        if self.components != None:
            self.components = t.cat((self.components, batch_scores), dim = 0)
        else:
            self.components = batch_scores 

        batch_scores = t.sigmoid(t.sum(batch_scores, dim = -1))

        if self.scores != None:
            self.scores = t.cat((self.scores, batch_scores), dim = 0)
        else:
            self.scores = batch_scores  

        del batch_scores
        t.cuda.empty_cache()

        t.save(self.components, comp_file)      

    def get_idiom_pos(self, aligned_positions, tags: list):
        """
        This method extract the position of the idiom.

        @params
            aligned_positions: aligned EPIE and model tokenizations
            tags: idiom labels
        """
        epie_idiom_pos = [i for i in range(len(tags)) if "IDIOM" in tags[i]]

        start = None
        end = None
        for epie_pos, model_positions in aligned_positions:
            if epie_pos == epie_idiom_pos[0]:
                start = model_positions[0]
            elif epie_pos == epie_idiom_pos[-1]:
                end = model_positions[-1]
        
        if end == None:
            end = aligned_positions[-1][-1][-1]
        assert start != None and end != None
        return (start, end)

    def align_tokens(self, sent: str, tokenized_sent: list, model_str_tokens: list):
        """
        This method aligns the tokenizations by EPIE and by the model.

        @params
            sent: natural sentence
            tokenized_sent: tokenization by EPIE
            model_str_tokens: tokenization by the model
        @returns
            list of the aligned token positions
        """
        aligned = self.aligner.align(
            types.TokenizedSet(tokens=[tokenized_sent, model_str_tokens], text=sent)
        )
        return list(aligned[0])
    
    def get_idiom_combinations(self, idiom_positions, sent_positions):
        '''
        This method extracts the positions of scores attending to idiom tokens.

        @params 
            idiom_positons: tensor with the positions of the idiom tokens
            sent_positions: tensor with the positions of all tokens in the sentence
        @returns idiom_combinations: list of position tuples of the scores attending to idiom tokens
        '''
        idiom_combinations = []
        for i in sent_positions.tolist():
            for j in sent_positions.tolist():
                if i >= j and (i in idiom_positions or j in idiom_positions):
                    idiom_combinations.append((i, j))
        return idiom_combinations
    
    def extract_tensor(self, attention_pattern, combined_positions: list):
        '''
        This method extracts the values of the given positions and stores them in a new tensor.

        @params
            attention_pattern: attention scores for one head and one sentence
            combined_positions: list of position tuples
        @returns tensor with the values of the given positions 
        '''
        return t.tensor([attention_pattern[combined_positions[i][0]][combined_positions[i][1]] for i in range(len(combined_positions))], dtype=t.float16, device=self.device)
    
    def max_idiom_toks(self, argmax_list: list, idiom_pos: tuple):
        '''
        This method computes the fraction of idiom tokens with a maximum score in a row or column of the attention pattern.

        @param 
            argmax_list: list with the tokens with the maximum score in a row/column
        @returns 
            idiom_frac: fraction of idiom tokens with the maximum score
        '''
        num_toks = len(argmax_list)
        max_counter = Counter(argmax_list)
        max_id, max_count = max_counter.most_common(1)[0]
        idiom_frac = 0.0
        if max_id >= idiom_pos[0] and max_id <= idiom_pos[1]:
            idiom_frac = max_count / num_toks
        else:
            idiom_frac = 0.0
        return idiom_frac
    
    def mean_qk_max(self, attention_pattern, idiom_pos):
        '''
        This method computes the mean fraction of idioms tokens having the highest score for the whole attention pattern.

        @param 
            attention_pattern: attention scores for one head and one sentence
        @returns 
            mean of the row and column fractions
        '''
        argmax_k = t.argmax(attention_pattern, dim=1)[idiom_pos[0]:]
        argmax_q = t.argmax(attention_pattern, dim=0)[:idiom_pos[1]+1]

        if argmax_k.size(0) == 0:
            q2k = t.zeros(1, device=self.device, dtype=t.float16)
        else:
            q2k = self.max_idiom_toks(argmax_k.tolist(), idiom_pos)
        
        if argmax_q.size(0) == 0:
            k2q = t.zeros(1, device=self.device, dtype=t.float16)
        else:
            k2q = self.max_idiom_toks(argmax_q.tolist(), idiom_pos)

        del argmax_k
        del argmax_q
        t.cuda.empty_cache()

        return (q2k + k2q)/2
    
    def compute_q_phrase_attention(self, attention_pattern, idiom_pos: tuple):
        '''
        This method computes the fraction of idiom tokens in the number_of_idiom_toks max scores per row.

        @param 
            attention_pattern: attention scores for one head and one sentence
            idiom_pos: start and end positions of the idiom
        @returns 
            mean fraction of idiom tokens in the top-k scores per row 
        '''
        sorted_tensor = t.sort(attention_pattern, dim=1, stable=True, descending = True)[1][idiom_pos[0]:]

        idiom_fractions = []
        num_idioms = idiom_pos[-1] - idiom_pos[0] + 1

        for pos, sorted_ids in enumerate(sorted_tensor.tolist()):
            if pos < num_idioms:
                num_idioms_row = pos + 1
            else:
                num_idioms_row = num_idioms

            max_indices = sorted_ids[:num_idioms_row]

            idiom_fraction = 0
            for max_index in max_indices:
                if max_index >= idiom_pos[0] and max_index <= idiom_pos[1]:
                    idiom_fraction += 1
            idiom_fractions.append(idiom_fraction / len(max_indices))

        if len(idiom_fractions) != 0:
            return sum(idiom_fractions)/len(idiom_fractions)
        else:
            return 0.0
    
    def compute_k_phrase_attention(self, attention_pattern, idiom_pos: tuple):
        '''
        This method computes the fraction of idiom tokens in the number_of_idiom_toks max scores per column.

        @param 
            attention_pattern: attention scores for one head and one sentence
            idiom_pos: start and end positions of the idiom
        @returns 
            mean fraction of idiom tokens in the top-idiom_len scores per column 
        '''
        sorted_tensor = t.sort(attention_pattern, dim=0, stable=True, descending = True)[1][:idiom_pos[1]+1]

        idiom_fractions = []
        num_idioms = idiom_pos[-1] - idiom_pos[0] + 1

        for pos, sorted_ids in enumerate(sorted_tensor.tolist()):
            if pos >= idiom_pos[0]:
                num_idioms_col = idiom_pos[1] - pos + 1
            else:
                num_idioms_col = num_idioms

            max_indices = sorted_ids[:num_idioms_col]

            idiom_fraction = 0
            for max_index in max_indices:
                if max_index >= idiom_pos[0] and max_index <= idiom_pos[1]:
                    idiom_fraction += 1
            idiom_fractions.append(idiom_fraction / len(max_indices))

        if len(idiom_fractions) != 0:
            return sum(idiom_fractions)/len(idiom_fractions)
        else:
            return 0.0
    
    def mean_qk_phrase(self, attention_pattern, idiom_pos: tuple):
        '''
        This method computes the mean of the maximum phrase attention scores for the whole attention matrix.

        @param 
            attention_pattern: attention scores for one head and one sentence
            idiom_pos: start and end positions of the idiom
        @returns 
            mean of the maximum phrase attention scores per row/column
        '''
        q2k = self.compute_q_phrase_attention(attention_pattern, idiom_pos)
        k2q = self.compute_k_phrase_attention(attention_pattern, idiom_pos)
        return (q2k + k2q)/2
    
    def compute_components(self, attention_pattern, head_result, idiom_pos: tuple):
        '''
        This method computes the components of the Idiom Score for one head and one sentence.

        @params 
            attention_pattern: attention scores for one head and one sentence
            head_result: attention result
            idiom_pos: start and end position of the idiom
        @returns 
            mean attention scores for the idiom, standard deviation of the mean, Single Max Score, Phrase Max Score, Contribution
        '''
        idiom_positions = t.arange(idiom_pos[0], idiom_pos[1]+1, device=self.device)
        sent_positions = t.arange(attention_pattern.size(-1)).to(self.device)

        idiom_attention = self.extract_tensor(attention_pattern, self.get_idiom_combinations(idiom_positions, sent_positions))
        max_tok = self.mean_qk_max(attention_pattern, idiom_pos)
        phrase = self.mean_qk_phrase(attention_pattern, idiom_pos)
        contribution = self.get_attn_contribution(head_result, idiom_pos)

        del idiom_positions
        del sent_positions
        t.cuda.empty_cache()

        return t.tensor((t.mean(idiom_attention), -t.std(idiom_attention), max_tok, phrase, contribution), dtype=t.float16, device = self.device)

    def save_tensor(self, tensor, filename: str):
        """
        This method saves the resulting tensor.

        @params
            tensor: tensor to save
            filename: file where the tensor is stored
        """
        t.save(tensor, filename)

    def explore_tensor(self):
        """
        Sanity check for the results.
        """
        print(f"The score of the first sentence for the layer 0 and head 0 is:\n{self.scores[0][0][0]}")

    def get_attn_contribution(self, head_result, idiom_pos: tuple):
        """
        This method computes the mean of the attention results for the idiom.

        @params
            head_result: attention result
            idiom_pos: start and end of the idiom
        @returns
            mean of the attention result for the idiom
        """
        # seq_len x dmodel
        return t.mean(head_result[idiom_pos[0]:idiom_pos[1]+1])
    

if __name__ == "__main__":
    model: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-14m")
    model.cfg.use_attn_result = True
    _, cache = model.run_with_cache("‘Are not you going to spill the beans?", remove_batch_dim = True)
    