from data import EPIE_Data
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
    def __init__(self, model, filename: str = "pythia_formal_idiom_pos.json", start = 0, cage_dir = ""):
        #self.data = EPIE_Data()
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
        for i in range(len(batch["sentence"])):
            sent = batch["sentence"][i]
            model_str_tokens= self.model.to_str_tokens(sent)
            aligned_positions = self.align_tokens(sent, batch["tokenized"][i], model_str_tokens)
            self.idiom_positions.append(self.get_idiom_pos(aligned_positions, batch["tags"][i]))

    def store_all_idiom_pos(self, filename):
        with open(filename, 'w', encoding = "utf-8") as f:
            json.dump(self.idiom_positions, f)    

    def load_all_idiom_pos(self, filename):
        if os.path.isfile(filename):
            with open(filename, 'r', encoding = "utf-8") as f:
                return json.load(f) 
        else:
            return [] 

    def get_cache(self, sent):
        idiom_tokens = self.model.to_tokens(sent)
        _, cache = self.model.run_with_cache(idiom_tokens, remove_batch_dim=True)
        return cache.to(self.device)

    def get_cage(self):
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


    def create_idiom_features(self, sent, idiom_pos):
        pattern_stack, result_stack = self.get_cage()
        cache = None
        if pattern_stack == None or result_stack == None:
            cache = self.get_cache(sent)

        layer_head_features = t.zeros(self.model.cfg.n_layers, self.model.cfg.n_heads, self.features, dtype=t.float16, device = self.device) # 8 features (4 idiom feats, 4 ngram feats)
        #activation_matrix = cache.stack_activation("pattern") # layers x heads x seq x seq
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

    def create_feature_tensor(self, sent, idiom_pos):
        cache = self.get_cache(sent)
        num_idioms = idiom_pos[-1] - idiom_pos[0] + 1

        model_str_tokens= self.model.to_str_tokens(sent)
        ngram_positions = self.get_ngram_pos(model_str_tokens, num_idioms, idiom_pos)

        layer_head_features = t.zeros(self.model.cfg.n_layers, self.model.cfg.n_heads, 8, dtype=t.float16, device = self.device) # 8 features (4 idiom feats, 4 ngram feats)
        #activation_matrix = cache.stack_activation("pattern") # layers x heads x seq x seq
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                attention_pattern = cache["pattern", layer][head].to(dtype=t.float16)
                idiom_features = self.compute_components(attention_pattern, idiom_pos)
                ngram_features = self.compute_ngram_features(attention_pattern, ngram_positions, idiom_features)

                layer_head_features[layer][head] = t.hstack((idiom_features, ngram_features))

                del attention_pattern
                del idiom_features
                del ngram_features
                t.cuda.empty_cache()
        
        del cache
        t.cuda.empty_cache()

        return layer_head_features
    
    def create_score_tensor(self, sent, idiom_pos):
        """ @deprecated, can use feature_tensor imidiately
        """
        feature_tensor = self.create_feature_tensor(sent, idiom_pos)

        return t.sigmoid(t.sum(feature_tensor, dim = -1))
    
    def create_idiom_score_tensor(self, batch, comp_file):
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

        #t.save(self.scores, ckp_file)    
        t.save(self.components, comp_file)    

    def create_data_score_tensor(self, batch, ckp_file):
        batch_scores = t.zeros(len(batch["sentence"]), self.model.cfg.n_layers, self.model.cfg.n_heads, 8, dtype=t.float16, device = self.device)

        for i in range(len(batch["sentence"])):
            batch_scores[i] = self.create_feature_tensor(batch["sentence"][i], batch["idiom_pos"][i])
        # NO COMPONENTS! 
        batch_scores = t.sigmoid(t.sum(batch_scores, dim = -1))

        if self.scores != None:
            self.scores = t.cat((self.scores, batch_scores), dim = 0)
        else:
            self.scores = batch_scores  

        del batch_scores
        t.cuda.empty_cache()

        t.save(self.scores, ckp_file)    


    def get_ngram_positions(self, num_idioms, tokenized_sent, aligned_positions, idiom_pos):
        """
        @deprecated Use get_ngram_pos instead
        """
        ngrams = self.create_ngrams(tokenized_sent, num_idioms, idiom_pos)
        ngram_positions = dict()
        for ngram in ngrams:
            ngram_positions[self.get_idiom_pos(aligned_positions, tokenized_sent)] = ngram
        return ngram_positions


    def compute_ngram_features(self, attention_pattern, ngram_positions, idiom_features):
        ngram_features = t.zeros(idiom_features.size(0), dtype=t.float16, device=self.device)
        #for ngram_pos, ngram in ngram_positions.items():
        for ngram_pos in ngram_positions:
            ngram_features = t.vstack((ngram_features, self.compute_components(attention_pattern, ngram_pos)))

        return t.sum(idiom_features >= ngram_features[1:], dim=0)/ngram_features[1:].size(0)


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
    
    def get_ngram_pos(self, model_str_tokens, num_idioms, idiom_pos):
        ngrams = []
        for pos in range(len(model_str_tokens)):
            if pos != idiom_pos[0] and pos <= (len(model_str_tokens)-num_idioms):
                ngrams.append((pos, pos+num_idioms-1))
        return ngrams

    def align_tokens(self, sent: str, tokenized_sent: list, model_str_tokens: list):
        aligned = self.aligner.align(
            types.TokenizedSet(tokens=[tokenized_sent, model_str_tokens], text=sent)
        )
        return list(aligned[0])
    
    def create_ngrams(self, model_str_tokens, num_idioms, idiom_pos):
        '''
        This method creates a list of the ngrams in a sentence.

        @returns ngrams: list of the ngrams in the sentence
        '''
        ngrams = dict()
        for pos in range(len(model_str_tokens)):
            if pos != idiom_pos[0] and pos <= (len(model_str_tokens)-num_idioms):
                toks = [tok.strip() for tok in model_str_tokens[pos:(pos+num_idioms)]]
                ngrams[(pos, pos+num_idioms-1)] = toks
        return ngrams
    
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
    
    def max_idiom_toks(self, argmax_list: list, idiom_pos):
        '''
        This method computes the fraction of idiom tokens with a maximum score in a row or column of the attention pattern.

        @param argmax_list: list with the tokens with the maximum score in a row/column
        @returns idiom_frac: fraction of idiom tokens with the maximum score
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

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean of the row and column fractions
        '''
        # q2k = self.max_idiom_toks(t.argmax(attention_pattern, dim=1).tolist()[idiom_pos[0]:], idiom_pos)
        # k2q = self.max_idiom_toks(t.argmax(attention_pattern, dim=0).tolist()[:idiom_pos[1]+1], idiom_pos)

        literal_argmax_k = t.argmax(attention_pattern, dim=1)[idiom_pos[0]:]
        literal_argmax_q = t.argmax(attention_pattern, dim=0)[:idiom_pos[1]+1]

        if literal_argmax_k.size(0) == 0:
            q2k = t.zeros(1, device=self.device, dtype=t.float16)
        else:
            q2k = self.max_idiom_toks(literal_argmax_k.tolist(), idiom_pos)
        
        if literal_argmax_q.size(0) == 0:
            k2q = t.zeros(1, device=self.device, dtype=t.float16)
        else:
            k2q = self.max_idiom_toks(literal_argmax_q.tolist(), idiom_pos)

        del literal_argmax_k
        del literal_argmax_q
        t.cuda.empty_cache()

        return (q2k + k2q)/2
    
    def compute_q_phrase_attention(self, attention_pattern, idiom_pos):
        '''
        This method computes the fraction of idiom tokens in the number_of_idiom_toks max scores per row.

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean fraction of idiom tokens in the top-idiom_len scores per row 
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
        #return sum(idiom_fractions)/len(idiom_fractions)
    
    def compute_k_phrase_attention(self, attention_pattern, idiom_pos):
        '''
        This method computes the fraction of idiom tokens in the number_of_idiom_toks max scores per column.

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean fraction of idiom tokens in the top-idiom_len scores per column 
        '''
        sorted_tensor = t.sort(attention_pattern, dim=0, stable=True, descending = True)[1][:idiom_pos[1]+1]

        idiom_fractions = []
        num_idioms = idiom_pos[-1] - idiom_pos[0] + 1

        for pos, sorted_ids in enumerate(sorted_tensor.tolist()):
            if pos >= idiom_pos[0]:
                num_idioms_row = idiom_pos[1] - pos + 1 # Name auf Spalte anpassen?
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
        #return sum(idiom_fractions)/len(idiom_fractions)
    
    def mean_qk_phrase(self, attention_pattern, idiom_pos):
        '''
        This method computes the mean of the maximum phrase attention scores for the whole attention matrix.

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean of the maximum phrase attention scores per row/column
        '''
        q2k = self.compute_q_phrase_attention(attention_pattern, idiom_pos)
        k2q = self.compute_k_phrase_attention(attention_pattern, idiom_pos)
        return (q2k + k2q)/2
    
    def compute_components(self, attention_pattern, head_result, idiom_pos):
        '''
        This method computes the components of the idiom score for one head and one sentence.

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean on the scores attending to idiom tokens, standard deviation of the mean, probability of an idiom token being the maximum score of the row/column, fraction of idiom tokens within the max idiom_len-scores
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

    def compute_mean_batched(self, batch, ckp_file):
        batch_scores = t.zeros(len(batch["sentence"]), self.model.cfg.n_layers, self.model.cfg.n_heads, 2, dtype=t.float16, device = self.device)

        layers = self.model.cfg.n_layers
        heads = self.model.cfg.n_heads

        for i in range(len(batch["sentence"])):
            sent = batch["sentence"][i]

            idiom_pos = batch["idiom_pos"][i]
            idiom_positions = t.arange(idiom_pos[0], idiom_pos[1]+1, device=self.device)

            cache = self.get_cache(sent)
            #activation_matrix = cache.stack_activation("pattern") # layers x heads x seq x seq
            for layer in range(layers):
                for head in range(heads):
                    attention_pattern = cache["pattern", layer][head]
                    sent_positions = t.arange(attention_pattern.size(-1)).to(self.device)
                    idiom_tensor = self.extract_tensor(attention_pattern, self.get_idiom_combinations(idiom_positions, sent_positions))
                    batch_scores[i][layer][head] = t.tensor((t.mean(idiom_tensor), -t.std(idiom_tensor)), dtype=t.float16, device=self.device)

            # n_idiom = idiom_pos[1] - idiom_pos[0] + 1

            # q_idiom = activation_matrix[:, :, idiom_pos[0]:idiom_pos[1]+1].reshape(layers, heads, n_idiom * activation_matrix.size(2))
            # k_idiom = activation_matrix[:, :, :, idiom_pos[0]:idiom_pos[1]+1].reshape(layers, heads, n_idiom * activation_matrix.size(2))
            # batch_scores[i] = t.mean(t.cat((q_idiom, k_idiom), dim = -1), dim = -1)
            # idiom_attention = self.extract_tensor(activation_matrix, self.get_idiom_combinations(idiom_positions, sent_positions))
        #batch_scores = t.sigmoid(t.sum(batch_scores, dim = -1))

        if self.scores != None:
            self.scores = t.cat((self.scores, batch_scores), dim = 0)
        else:
            self.scores = batch_scores  

        t.save(self.scores, ckp_file)

    def create_component_table(self):
        '''
        This method creates a dataframe containing the score components
        '''
        layer_heads = [f"{layer}.{head}" for layer in range(self.model.cfg.n_layers) for head in range(self.model.cfg.n_heads)]

        ngrams = self.create_ngrams()

        means = []
        mean_diff = []
        stds = []
        std_diff = []
        p_max = []
        p_max_diff = []
        p_phrase = []
        p_phrase_diff = []

        for layer_head in layer_heads:
            layer, head = layer_head.split('.')

            attention_pattern = self.cache["pattern", int(layer)][int(head)]
            idiom_mean, idiom_std, idiom_max, idiom_phrase = self.compute_components(attention_pattern, (self.start_idiom_pos, self.end_idiom_pos))

            means.append(idiom_mean)
            stds.append(idiom_std)
            p_max.append(idiom_max)
            p_phrase.append(idiom_phrase)

            rel_mean = 0
            rel_std = 0
            rel_max = 0
            rel_phrase = 0
            for ngram in ngrams:
                ngram_pos = self.get_idiom_pos(ngram)
                ngram_mean, ngram_std, ngram_max, ngram_phrase = self.compute_components(attention_pattern, ngram_pos)

                if idiom_mean >= ngram_mean:
                    rel_mean += 1

                if idiom_std <= ngram_std:
                    rel_std += 1

                if idiom_max >= ngram_max:
                    rel_max += 1

                if idiom_phrase >= ngram_phrase:
                    rel_phrase += 1

            num_ngrams = len(ngrams)
            mean_diff.append(rel_mean/num_ngrams)
            std_diff.append(rel_std/num_ngrams)
            p_max_diff.append(rel_max/num_ngrams)
            p_phrase_diff.append(rel_phrase/num_ngrams)

        df = pd.DataFrame(dict(
            attention_heads_per_layer = layer_heads,
            mean = means,
            relative_means = mean_diff,
            std = stds,
            relative_stds = std_diff,
            p_max = p_max,
            relative_p_max = p_max_diff,
            p_phrase = p_phrase,
            relative_p_phrase = p_phrase_diff
        ))
        return df
    
    def compute_idiom_score(self, sent, tokenized_sent, tags):
        '''
        This method aggregates the components of the scores and brings them between 0 and 1.

        @params
            w: weights for the components
            b: bias factor
        '''
        layer_heads = [f"{layer}.{head}" for layer in range(self.model.cfg.n_layers) for head in range(self.model.cfg.n_heads)]
        score_per_head = defaultdict(float)

        self.str_sent = sent
        self.str_tokens = self.model.to_str_tokens(self.str_sent)

        sigmoid = t.nn.Sigmoid()

        cache = self.get_cache(self.str_sent)

        self.start_idiom_pos, self.end_idiom_pos = self.match_idiom_pos(tokenized_sent, tags)
        #self.start_idiom_pos, self.end_idiom_pos = self.get_new_idiom_pos(self.data.get_idiom_toks(tokenized_sent.split(' '), tags.split(' ')))
        self.num_idioms = self.end_idiom_pos - self.start_idiom_pos + 1

        ngrams = self.create_ngrams(tokenized_sent.split(' '))
        num_ngrams = len(ngrams)

        for layer_head in layer_heads:
            layer, head = layer_head.split('.')

            attention_pattern = cache["pattern", int(layer)][int(head)]
            idiom_mean, idiom_std, idiom_max, idiom_phrase = self.compute_components(attention_pattern, (self.start_idiom_pos, self.end_idiom_pos))

            rel_mean = 0
            rel_std = 0
            rel_max = 0
            rel_phrase = 0
            for ngram in ngrams:
                ngram_pos = self.get_idiom_pos(ngram)
                ngram_mean, ngram_std, ngram_max, ngram_phrase = self.compute_components(attention_pattern, ngram_pos)

                if idiom_mean >= ngram_mean:
                    rel_mean += 1

                if idiom_std <= ngram_std:
                    rel_std += 1

                if idiom_max >= ngram_max:
                    rel_max += 1

                if idiom_phrase >= ngram_phrase:
                    rel_phrase += 1

            mean_diff = rel_mean/num_ngrams
            std_diff = rel_std/num_ngrams
            p_max_diff = rel_max/num_ngrams
            p_phrase_diff = rel_phrase/num_ngrams

            # TODO: Add weights and bias (not wandb)
            score_per_head[layer_head] = float(sigmoid(t.tensor(idiom_mean-idiom_std + idiom_max + idiom_phrase + mean_diff + std_diff + p_max_diff + p_phrase_diff)))

        return score_per_head

    def compute_scores_per_head(self, sents, tokenized_sents, tags):
        scores_per_head = defaultdict(list)

        for pos, sent in tqdm(enumerate(sents), desc="Processing idiom sentences"):
            tokenized_sent = tokenized_sents[pos]
            sent_tags = tags[pos] 
            score_per_head = self.compute_idiom_score(sent, tokenized_sent, sent_tags)

            for head, score in score_per_head.items():
                scores_per_head[head].append(score)
        return scores_per_head

    def get_avg_idiom_scores(self, scores_per_head):
        avg_score_per_head = defaultdict(float)

        num_sents = len(scores_per_head["0.0"])
        avg_score_per_head = {head: (sum(scores)/num_sents) for head, scores in avg_score_per_head.items()}
        self.explore_scores(avg_score_per_head)
        return avg_score_per_head

    def save_scores(self, score_per_head, filename):
        with open(filename, 'w', encoding = "utf-8") as f:
            json.dump(score_per_head, f)

    def save_tensor(self, tensor, filename):
        t.save(tensor, filename)

    def explore_scores(score_per_head):
        print(f"Maximum head: {max(score_per_head, key=lambda k:score_per_head.get(k))} - {max(score_per_head.values())}")
        print(f"Minimum head: {min(score_per_head, key=lambda k:score_per_head.get(k))} - {min(score_per_head.values())}")

        print(f"Sorted highest to lowest: {[(head, score_per_head.get(head)) for head in sorted(score_per_head, key = lambda k:score_per_head.get(k), reverse = True)]}")
        print(f"Sorted lowest to highest: {[(head, score_per_head.get(head)) for head in sorted(score_per_head, key = lambda k:score_per_head.get(k))]}")

    def explore_tensor(self):
        print(f"The score of the first sentence for the layer 0 and head 0 is:\n{self.scores[0][0][0]}")

    def get_attention_contributions(
        self, 
        resid_pre,#: t.Tensor #"batch pos d_model"
        resid_mid,#: Float[torch.Tensor, "batch pos d_model"],
        decomposed_attn,#: Float[torch.Tensor, "batch pos key_pos head d_model"],
        distance_norm: int = 1,
    ): #-> Tuple[
       # Float[torch.Tensor, "batch pos key_pos head"],
        #Float[torch.Tensor, "batch pos"],]
        """
        AUS LLMTransparencyTOOL
        Returns a pair of
        - a tensor of contributions of each token via each head
        - the contribution of the residual stream.
        """

        # part dimensions | batch dimensions | vector dimension
        # ----------------+------------------+-----------------
        # key_pos, head   | batch, pos       | d_model
        parts = einops.rearrange(
            decomposed_attn,
            "batch pos key_pos head d_model -> key_pos head batch pos d_model",
        )
        attn_contribution, residual_contribution = self.get_contributions_with_one_off_part(
            parts, resid_pre, resid_mid, distance_norm
        )
        return (
            einops.rearrange(
                attn_contribution, "key_pos head batch pos -> batch pos key_pos head"
            ),
            residual_contribution,
        )
    
    def get_contributions_with_one_off_part(
        self,
        parts,#: torch.Tensor,
        one_off,#: torch.Tensor,
        whole,#: torch.Tensor,
        distance_norm: int = 1,
        ): #-> Tuple[torch.Tensor, torch.Tensor]:
        """
        AUS LLM TransparencyTOOL
        Same as computing the contributions, but there is one additional part. That's useful
        because we always have the residual stream as one of the parts.

        See `get_contributions` documentation about `parts` and `whole` dimensions. The
        `one_off` should have the same dimensions as `whole`.

        Returns a pair consisting of
        1. contributions tensor for the `parts`
        2. contributions tensor for the `one_off` vector
        """
        assert one_off.shape == whole.shape

        k = len(parts.shape) - len(whole.shape)
        assert k >= 0

        # Flatten the p_ dimensions, get contributions for the list, unflatten.
        flat = parts.flatten(start_dim=0, end_dim=k - 1)
        flat = t.cat([flat, one_off.unsqueeze(0)])
        contributions = self.get_contributions(flat, whole, distance_norm)
        parts_contributions, one_off_contributions = t.split(
            contributions, flat.shape[0] - 1
        )
        return (
            parts_contributions.unflatten(0, parts.shape[0:k]),
            one_off_contributions[0],
        )
    
    def get_contributions(
        self, 
        parts,#: torch.Tensor,
        whole,#: torch.Tensor,
        distance_norm: int = 1,
    ):# -> torch.Tensor:
        """
        AUS LLMTRANSPARENCYTOOL
        Compute contributions of the `parts` vectors into the `whole` vector.

        Shapes of the tensors are as follows:
        parts:  p_1 ... p_k, v_1 ... v_n, d
        whole:               v_1 ... v_n, d
        result: p_1 ... p_k, v_1 ... v_n

        Here
        * `p_1 ... p_k`: dimensions for enumerating the parts
        * `v_1 ... v_n`: dimensions listing the independent cases (batching),
        * `d` is the dimension to compute the distances on.

        The resulting contributions will be normalized so that
        for each v_: sum(over p_ of result(p_, v_)) = 1.
        """
        EPS = 1e-5

        k = len(parts.shape) - len(whole.shape)
        assert k >= 0
        assert parts.shape[k:] == whole.shape
        bc_whole = whole.expand(parts.shape)  # new dims p_1 ... p_k are added to the front

        distance = t.nn.functional.pairwise_distance(parts, bc_whole, p=distance_norm)

        whole_norm = t.norm(whole, p=distance_norm, dim=-1)
        distance = (whole_norm - distance).clip(min=EPS)

        sum = distance.sum(dim=tuple(range(k)), keepdim=True)

        return distance / sum

    def get_attn_contribution(self, head_result, idiom_pos):
        # seq_len x dmodel
        return t.mean(head_result[idiom_pos[0]:idiom_pos[1]+1])
    

if __name__ == "__main__":
    model: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-14m")
    # epie = EPIE_Data()
    # scorer = IdiomScorer(model)

    # data = epie.create_hf_dataset(epie.formal_sents, epie.tokenized_formal_sents, epie.tags_formal)
    # data.map(lambda batch: scorer.get_all_idiom_pos(batch), batched = True)
    # print(len(scorer.idiom_positions))
    # scorer.store_all_idiom_pos("formal")
    # formal_scores = scorer.create_data_score_tensor(formal_data)

    # for i in range(len(formal_data)):
    #     model_str_tokens = model.to_str_tokens(formal_data["sentence"][i])
    #     aligned_positions = scorer.align_tokens(formal_data["sentence"][i], formal_data["tokenized"][i], model_str_tokens)
    #     idiom_pos = scorer.get_idiom_pos(aligned_positions, formal_data["tags"][i])

    # loaded_scores = t.load("./scores/test_formal.pt", weights_only = False).to(scorer.device)
    # #assert(len(loaded_scores) == len(formal_data))  
    # scorer.explore_tensor(loaded_scores)

    #detect_scores = head_detector.detect_head(model, "‘Are not you going to spill the beans?", t.tril(t.ones(11, 11)))
    model.cfg.use_attn_result = True
    _, cache = model.run_with_cache("‘Are not you going to spill the beans?", remove_batch_dim = True)
    head_detector.compute_head_attention_similarity_score(cache["pattern", 0][0], t.tril(t.zeros(11, 11)), exclude_bos = False, error_measure="abs")
    # print(detect_scores)
    # print(detect_scores.size())
    