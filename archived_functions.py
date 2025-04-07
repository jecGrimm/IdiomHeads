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