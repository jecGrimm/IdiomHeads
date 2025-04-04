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

class Scorer:
    def __init__(self, model):
        self.data = EPIE_Data()
        self.model = model
        self.aligner = PythonGreedyCoverageAligner()

    def get_cache(self, sent):
        idiom_tokens = self.model.to_tokens(sent)
        logits, cache = self.model.run_with_cache(idiom_tokens, remove_batch_dim=True)
        return cache
    
    # def get_idiom_pos(self, idiom_toks):
    #     '''
    #     This method extracts the start and end position of the idiom in the tokenized sentence of the model.

    #     @asserts if the joined tokens are equal to the joined idiom string in the EPIE dataset
    #     @returns (start, end): tuple of the start and end position of the idiom
    #     '''
    #     #wo_eot_tokens = self.model.to_str_tokens(sent, prepend_bos = False)

    #     idiom_str = self.data.remove_spaces([' '.join(idiom_toks)])[0]
    #     sent_idx = self.str_sent.index(idiom_str) - 1

    #     if sent_idx > -1:
    #         idiom_str = self.str_sent[sent_idx] + idiom_str

    #         model_idiom_toks = self.model.to_str_tokens(idiom_str, prepend_bos=False)
    #         first_tok = model_idiom_toks[0]
    #         if first_tok.strip() not in idiom_str[1:]:
    #             model_idiom_toks = model_idiom_toks[1:]
    #             idiom_str = idiom_str[1:]
    #     else:
    #         model_idiom_toks = self.model.to_str_tokens(idiom_str, prepend_bos=False)

    #     if model_idiom_toks[0] not in self.str_tokens:
    #         model_idiom_toks = model_idiom_toks[1:]
    #         idiom_str = idiom_str[1:]

    #     start = self.str_tokens.index(model_idiom_toks[0])
    #     end = start+len(model_idiom_toks)-1

    #     while not all((y in x or x in y) for x, y in zip(self.str_tokens[start:end+1], model_idiom_toks)):
    #         start = start + self.str_tokens[start+1:].index(model_idiom_toks[0]) + 1
    #         end = start + len(model_idiom_toks) - 1

    #     #assert ''.join(self.str_tokens[start:end+1]).strip() == idiom_str.strip(), f"\nEPIE Tokens: {idiom_toks}\nExtracted Tokens: {self.str_tokens[start:end+1]}"
    #     return (start, end)

    def get_new_idiom_pos(self, sent: str, tokenized_sent: list, model_str_tokens: list, tags: list):
        epie_idiom_pos = [i for i in range(len(tags)) if "IDIOM" in tags[i]]
        aligned_positions = self.align_tokens(sent, tokenized_sent, model_str_tokens)

        start = None
        end = None
        for epie_pos, model_positions in aligned_positions:
            if epie_pos == epie_idiom_pos[0]:
                start = model_positions[0]
            elif epie_pos == epie_idiom_pos[-1]:
                end = model_positions[-1]
        assert(start and end)
        return (start, end)


    # def match_idiom_pos(self, tokenized_sent, tags):
    #     tags = tags.split()
    #     epie_idiom_pos = [i for i in range(len(tags)) if "IDIOM" in tags[i]]
        
    #     pos_matches = defaultdict(int)
    #     model_pos = None
    #     for epie_pos, epie_tok in enumerate(tokenized_sent.split()):
    #         if epie_pos == 0:
    #             model_pos = 1

    #         if model_pos < len(self.str_tokens):
    #             model_tok = self.str_tokens[model_pos]

    #             if epie_tok == model_tok.strip():
    #                 pos_matches[epie_pos] = model_pos
    #                  model_pos += 1
    #             else:
    #                 if model_tok.strip() in epie_tok:
    #                     i = 1
    #                     curr_tok = model_tok
    #                     while model_pos + i < len(self.str_tokens):
    #                         curr_tok += self.str_tokens[model_pos + i].strip()
                            
    #                         if curr_tok == epie_tok:
    #                             pos_matches[epie_pos] = model_pos+i
    #                             model_pos = model_pos + i + 1
    #                             i = len(self.str_tokens)
        
    #     assert(''.join(tokenized_sent.split()[epie_idiom_pos[0]:epie_idiom_pos[-1]+1]) == ' '.join(self.str_tokens[pos_matches[epie_idiom_pos[0]]: pos_matches[epie_idiom_pos[-1]]+1]).replace(' ', ''))
    #     return pos_matches[epie_idiom_pos[0]], pos_matches[epie_idiom_pos[-1]]

    def align_tokens(self, sent: str, tokenized_sent: list, model_str_tokens: list):
        aligned = self.aligner.align(
            types.TokenizedSet(tokens=[tokenized_sent, model_str_tokens], text=sent)
        )
        return list(aligned[0])

    
    def create_ngrams(self, tokenized_sent):
        '''
        This method creates a list of the ngrams in a sentence.

        @returns ngrams: list of the ngrams in the sentence
        '''
        ngrams = []
        for pos in range(len(tokenized_sent)):
            if pos != self.start_idiom_pos and pos <= (len(tokenized_sent)-self.num_idioms):
                toks = [tok.strip() for tok in tokenized_sent[pos:(pos+self.num_idioms)]]
                ngrams.append(toks)
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
        return t.tensor([attention_pattern[combined_positions[i][0]][combined_positions[i][1]] for i in range(len(combined_positions))])
    
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
        q2k = self.max_idiom_toks(t.argmax(attention_pattern, dim=1).tolist()[idiom_pos[0]:], idiom_pos)
        k2q = self.max_idiom_toks(t.argmax(attention_pattern, dim=0).tolist()[:idiom_pos[1]+1], idiom_pos)

        return (q2k + k2q)/2
    
    def compute_q_phrase_attention(self, attention_pattern, idiom_pos):
        '''
        This method computes the fraction of idiom tokens in the number_of_idiom_toks max scores per row.

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean fraction of idiom tokens in the top-idiom_len scores per row 
        '''
        sorted_tensor = t.sort(attention_pattern, dim=1, stable=True, descending = True)[1][idiom_pos[0]:]

        idiom_fractions = []

        for pos, sorted_ids in enumerate(sorted_tensor.tolist()):
            if pos < self.num_idioms:
                num_idioms_row = pos + 1
            else:
                num_idioms_row = self.num_idioms

            max_indices = sorted_ids[:num_idioms_row]

            idiom_fraction = 0
            for max_index in max_indices:
                if max_index >= idiom_pos[0] and max_index <= idiom_pos[1]:
                    idiom_fraction += 1
            idiom_fractions.append(idiom_fraction / len(max_indices))

        return sum(idiom_fractions)/len(idiom_fractions)
    
    def compute_k_phrase_attention(self, attention_pattern, idiom_pos):
        '''
        This method computes the fraction of idiom tokens in the number_of_idiom_toks max scores per column.

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean fraction of idiom tokens in the top-idiom_len scores per column 
        '''
        sorted_tensor = t.sort(attention_pattern, dim=0, stable=True, descending = True)[1][:idiom_pos[1]+1]

        idiom_fractions = []

        for pos, sorted_ids in enumerate(sorted_tensor.tolist()):
            if pos >= idiom_pos[0]:
                num_idioms_row = idiom_pos[1] - pos + 1 # Name auf Spalte anpassen?
            else:
                num_idioms_row = self.num_idioms

            max_indices = sorted_ids[:num_idioms_row]

            idiom_fraction = 0
            for max_index in max_indices:
                if max_index >= idiom_pos[0] and max_index <= idiom_pos[1]:
                    idiom_fraction += 1
            idiom_fractions.append(idiom_fraction / len(max_indices))

        return sum(idiom_fractions)/len(idiom_fractions)
    
    def mean_qk_phrase(self, attention_pattern, idiom_pos):
        '''
        This method computes the mean of the maximum phrase attention scores for the whole attention matrix.

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean of the maximum phrase attention scores per row/column
        '''
        q2k = self.compute_q_phrase_attention(attention_pattern, idiom_pos)
        k2q = self.compute_k_phrase_attention(attention_pattern, idiom_pos)
        return (q2k + k2q)/2
    
    def compute_components(self, attention_pattern, idiom_pos):
        '''
        This method computes the components of the idiom score for one head and one sentence.

        @param attention_pattern: attention scores for one head and one sentence
        @returns mean on the scores attending to idiom tokens, standard deviation of the mean, probability of an idiom token being the maximum score of the row/column, fraction of idiom tokens within the max idiom_len-scores
        '''
        idiom_positions = t.arange(idiom_pos[0], idiom_pos[1]+1)
        sent_positions = t.arange(attention_pattern.size(-1))

        idiom_attention = self.extract_tensor(attention_pattern, self.get_idiom_combinations(idiom_positions, sent_positions))
        max_tok = self.mean_qk_max(attention_pattern, idiom_pos)
        phrase = self.mean_qk_phrase(attention_pattern, idiom_pos)

        return float(t.mean(idiom_attention)), float(t.std(idiom_attention)), max_tok, phrase

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
                ngram_pos = self.get_new_idiom_pos(ngram)
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

            # TODO: Add weights and bias
            score_per_head[layer_head] = float(sigmoid(t.tensor(idiom_mean-idiom_std + idiom_max + idiom_phrase + mean_diff + std_diff + p_max_diff + p_phrase_diff)))

        return score_per_head

    def get_avg_idiom_scores(self, all_idiom_sents, all_tokenized_sents, all_tags):
        avg_score_per_head = defaultdict(float)

        for pos, sent in tqdm(enumerate(all_idiom_sents), desc="Processing idiom sentences"):
            tokenized_sent = all_tokenized_sents[pos]
            tags = all_tags[pos] 
            score_per_head = self.compute_idiom_score(sent, tokenized_sent, tags)

            for head, score in score_per_head.items():
                avg_score_per_head[head] += score

        num_sents = len(all_idiom_sents)
        avg_score_per_head = {head: (score/num_sents) for head, score in avg_score_per_head.items()}
        self.explore_scores(avg_score_per_head)
        return avg_score_per_head

    def save_scores(self, score_per_head, filename):
        with open(filename, 'w', encoding = "utf-8") as f:
            json.dump(score_per_head, f)

    def explore_scores(score_per_head):
        print(f"Maximum head: {max(score_per_head, key=lambda k:score_per_head.get(k))} - {max(score_per_head.values())}")
        print(f"Minimum head: {min(score_per_head, key=lambda k:score_per_head.get(k))} - {min(score_per_head.values())}")

        print(f"Sorted highest to lowest: {[(head, score_per_head.get(head)) for head in sorted(score_per_head, key = lambda k:score_per_head.get(k), reverse = True)]}")
        print(f"Sorted lowest to highest: {[(head, score_per_head.get(head)) for head in sorted(score_per_head, key = lambda k:score_per_head.get(k))]}")

if __name__ == "__main__":
    model: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-14m")
    epie = EPIE_Data()
    scorer = Scorer(model)

    #scorer.compute_idiom_score(epie.formal_sents[50], epie.tokenized_formal_sents[50], epie.tags_formal[50])

    print(scorer.get_new_idiom_pos(epie.formal_sents[50], epie.tokenized_formal_sents[50].split(), model.to_str_tokens(epie.formal_sents[50]), epie.tags_formal[50].split()))