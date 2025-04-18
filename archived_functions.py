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

# def logit_attribution(embed, l_results, W_U, tokens) -> t.Tensor:
#     '''
#     Inputs:
#         embed (seq_len, d_model): the embeddings of the tokens (i.e. token + position embeddings)
#         l1_results (seq_len, n_heads, d_model): the outputs of the attention heads at layer 1 (with head as one of the dimensions)
#         l2_results (seq_len, n_heads, d_model): the outputs of the attention heads at layer 2 (with head as one of the dimensions)
#         W_U (d_model, d_vocab): the unembedding matrix
#     Returns:
#         Tensor of shape (seq_len-1, n_components)
#         represents the concatenation (along dim=-1) of logit attributions from:
#             the direct path (position-1,1)
#             layer 0 logits (position-1, n_heads)
#             and layer 1 logits (position-1, n_heads)
#     '''
#     W_U_correct_tokens = W_U[:, tokens[1:]]

#     direct_attributions = einsum("emb seq, seq emb -> seq", W_U_correct_tokens, embed[:-1])
#     concats = [direct_attributions.unsqueeze(-1)]

#     for l_result in l_results:
#         concats.append(einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l_result[:-1]))
#     # l1_attributions = einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l1_results[:-1])
#     # l2_attributions = einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l2_results[:-1])
#     #return t.concat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)
#     return t.concat(concats, dim=-1)