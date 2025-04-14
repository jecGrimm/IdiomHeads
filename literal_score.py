from data import EPIE_Data
import torch as t 
from collections import Counter, defaultdict
import pandas as pd
import json
from tqdm import tqdm
from transformer_lens import (
    HookedTransformer,
)
from merge_tokenizers import PythonGreedyCoverageAligner, types
import os

class LiteralScorer:
    def __init__(self, model, split: str = "formal"):
        self.model = model
        self.aligner = PythonGreedyCoverageAligner()
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.scores = None
        self.idiom_positions = self.load_all_idiom_pos(split)

    def load_all_idiom_pos(self, split):
        if os.path.isfile(f"{split}_idiom_pos.json"):
            with open(f"{split}_idiom_pos.json", 'r', encoding = "utf-8") as f:
                return json.load(f) 
        else:
            return [] 

    def create_data_score_tensor(self, batch, ckp_file):
        batch_scores = t.zeros(len(batch["sentence"]), self.model.cfg.n_layers, self.model.cfg.n_heads, 4, dtype=t.float16, device = self.device)

        for i in range(len(batch["sentence"])):
            batch_scores[i] = self.create_feature_tensor(batch["sentence"][i], batch["idiom_pos"][i])
        batch_scores = t.sigmoid(t.sum(batch_scores, dim = -1))

        if self.scores != None:
            self.scores = t.cat((self.scores, batch_scores), dim = 0)
        else:
            self.scores = batch_scores  
        
        del batch_scores
        t.cuda.empty_cache()

        t.save(self.scores, ckp_file)  

    def create_feature_tensor(self, sent, idiom_pos):
        cache = self.get_cache(sent)

        layer_head_features = t.zeros(self.model.cfg.n_layers, self.model.cfg.n_heads, 4, dtype=t.float16, device = self.device) # 4 features (4 idiom feats)
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                attention_pattern = cache["pattern", layer][head].to(dtype=t.float16)
                layer_head_features[layer][head] = self.compute_components(attention_pattern, idiom_pos)

                del attention_pattern
                t.cuda.empty_cache()

        del cache
        t.cuda.empty_cache()

        return layer_head_features
    
    def get_cache(self, sent):
        token_ids = self.model.to_tokens(sent)
        _, cache = self.model.run_with_cache(token_ids, remove_batch_dim=True)
        return cache.to(self.device)
    
    def align_tokens(self, sent: str, tokenized_sent: list, model_str_tokens: list):
        aligned = self.aligner.align(
            types.TokenizedSet(tokens=[tokenized_sent, model_str_tokens], text=sent)
        )
        return list(aligned[0])
    
    def get_idiom_pos(self, aligned_positions, tags: list):
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
    
    def compute_components(self, attention_pattern, idiom_pos):
        '''
        This method computes the components of the idiom score for one head and one sentence.

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean on the scores attending to idiom tokens, standard deviation of the mean, probability of an idiom token being the maximum score of the row/column, fraction of idiom tokens within the max idiom_len-scores
        '''
        idiom_positions = t.arange(idiom_pos[0], idiom_pos[1]+1, device=self.device)
        sent_positions = t.arange(attention_pattern.size(-1)).to(self.device)

        literal_attention = self.extract_tensor(attention_pattern, self.get_literal_combinations(idiom_positions, sent_positions))
        max_tok = self.mean_qk_max(attention_pattern, idiom_pos)
        phrase = self.mean_qk_phrase(attention_pattern, idiom_pos)

        del idiom_positions
        del sent_positions
        t.cuda.empty_cache()

        return t.tensor((t.mean(literal_attention), -t.std(literal_attention), max_tok, phrase), dtype=t.float16, device = self.device)
    
    def get_literal_combinations(self, idiom_positions, sent_positions):
        '''
        This method extracts the positions of scores attending to idiom tokens.

        @params 
            idiom_positons: tensor with the positions of the idiom tokens
            sent_positions: tensor with the positions of all tokens in the sentence
        @returns idiom_combinations: list of position tuples of the scores attending to idiom tokens
        '''
        return [(i, j) for i in sent_positions for j in sent_positions if i >= j and i not in idiom_positions and j not in idiom_positions]
    
    def extract_tensor(self, attention_pattern, combined_positions: list):
        '''
        This method extracts the values of the given positions and stores them in a new tensor.

        @params
            attention_pattern: attention scores for one head and one sentence
            combined_positions: list of position tuples
        @returns tensor with the values of the given positions 
        '''
        return t.tensor([attention_pattern[combined_positions[i][0]][combined_positions[i][1]] for i in range(len(combined_positions))], dtype=t.float16, device=self.device)
    
    def mean_qk_max(self, attention_pattern, idiom_pos):
        '''
        This method computes the mean fraction of idioms tokens having the highest score for the whole attention pattern.

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean of the row and column fractions
        '''
        literal_argmax_k = t.cat((t.argmax(attention_pattern, dim = 1)[:9], t.argmax(attention_pattern, dim = 1)[13:])) # check device
        literal_argmax_q = t.cat((t.argmax(attention_pattern, dim = 0)[:9], t.argmax(attention_pattern, dim = 1)[13:]))
        q2k = self.max_idiom_toks(literal_argmax_k.tolist(), idiom_pos)
        k2q = self.max_idiom_toks(literal_argmax_q.tolist(), idiom_pos)

        del literal_argmax_k
        del literal_argmax_q
        t.cuda.empty_cache()

        return (q2k + k2q)/2
    
    def max_idiom_toks(self, argmax_list: list, idiom_pos):
        '''
        This method computes the fraction of idiom tokens with a maximum score in a row or column of the attention pattern.

        @param argmax_list: list with the tokens with the maximum score in a row/column
        @returns idiom_frac: fraction of idiom tokens with the maximum score
        '''
        num_toks = len(argmax_list)
        max_counter = Counter(argmax_list)
        max_id, max_count = max_counter.most_common(1)[0]
        literal_frac = 0.0
        if max_id < idiom_pos[0] or max_id > idiom_pos[1]:
            literal_frac = max_count / num_toks
        else:
            literal_frac = 0.0
        return literal_frac
    
    def mean_qk_phrase(self, attention_pattern, idiom_pos):
        '''
        This method computes the mean of the maximum phrase attention scores for the whole attention matrix.

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean of the maximum phrase attention scores per row/column
        '''
        q2k = self.compute_phrase_attention(attention_pattern, idiom_pos, dim = 1)
        k2q = self.compute_phrase_attention(attention_pattern, idiom_pos, dim = 0)
        return (q2k + k2q)/2
    
    def compute_phrase_attention(self, attention_pattern, idiom_pos, dim):
        '''
        This method computes the fraction of idiom tokens in the number_of_idiom_toks max scores per row.

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean fraction of idiom tokens in the top-idiom_len scores per row 
        '''
        sorted_tensor = t.sort(attention_pattern, dim=dim, stable=True, descending = True)[1]
        literal_fractions = []
        num_idioms = idiom_pos[-1] - idiom_pos[0] + 1

        for pos, sorted_ids in enumerate(sorted_tensor.tolist()):
            if pos < idiom_pos[0]:
                num_idioms_row = 0
            elif pos >= idiom_pos[1]:
                num_idioms_row = num_idioms
            else:
                num_idioms_row = pos - idiom_pos[0] + 1

            max_indices = t.tensor(sorted_ids[:pos-num_idioms_row+1], device = self.device)

            if len(max_indices) != 0:
                literal_fraction = 0
                for max_index in max_indices:
                    if max_index < idiom_pos[0] or max_index > idiom_pos[1]:
                        literal_fraction += 1
                literal_fractions.append(literal_fraction / len(max_indices))
            
            del max_indices
            t.cuda.empty_cache()

        del sorted_tensor
        t.cuda.empty_cache()

        if len(literal_fractions) != 0:
            return sum(literal_fractions)/len(literal_fractions)
        else:
            return 0.0
        
    def explore_tensor(self):
        print(f"The score of the first sentence for the layer 0 and head 0 is:\n{self.scores[0][0][0]}")

    
if __name__ == "__main__":
    model: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-14m")
    epie = EPIE_Data()
    tensor = t.randn(17, 17)
    scorer = LiteralScorer(model)
    scorer.compute_phrase_attention(tensor, (9, 12), dim = 1)