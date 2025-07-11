import torch as t
import os
import json
from merge_tokenizers import PythonGreedyCoverageAligner, types

class LogitAttribution:
    def __init__(self, model, filename: str = "pythia_formal_idiom_pos_dla.json"):
        """
        This class computes the DLA scores.

        @params
            model: examined model
            filename: file with the idiom positions
        """
        self.model = model
        self.model.cfg.use_attn_result = True
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.token_attr = None
        self.split_attr = None
        self.idiom_positions = self.load_all_idiom_pos(filename)
        self.labels = None
        self.cache = None
        self.residual_stack = None
        self.aligner = PythonGreedyCoverageAligner()

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

    def store_all_idiom_pos(self, filename: str):
        """
        This method saves the idiom positions.

        @params
          filename: output file
        """
        with open(filename, 'w', encoding = "utf-8") as f:
            json.dump(self.idiom_positions, f)   

    def get_cache(self, sent: str):
        """
        This method creates the activation cache for a sentence.

        @params
            sent: processed sentence
        @returns
            activation cache on the device
        """
        _, self.cache = self.model.run_with_cache(sent, remove_batch_dim=True)
        self.cache.to(self.device)

    def split_logit_attribution(self, logit_attr, idiom_pos: tuple):
        """
        This method groups the logit attribution by idiom and literal tokens.

        @params
            logit_attr: DLA scores for the sentence
            idiom_pos: start and end position of the idiom
        @returns
            stacked grouped DLA
        """
        idiom_attr = self.mean_idiom_attribution(logit_attr, idiom_pos)
        literal_attr = self.mean_literal_attribution(logit_attr, idiom_pos)
        return t.vstack((idiom_attr, literal_attr))

    def mean_idiom_attribution(self, logit_attr, idiom_pos: tuple):
        """
        This method computes the averga DLA for the idiom.

        @params
            logit_attr: DLA scores for the sentence
            idiom_pos: start and end position of the idiom
        @returns
            mean DLA of the idiom
        """
        idiom_tensor = logit_attr[idiom_pos[0]:idiom_pos[1]+1]
        if idiom_tensor.size(0) == 0:
            return t.zeros(logit_attr.size(1), dtype=t.float16, device = self.device)
        else:
            return t.mean(idiom_tensor, dim = 0)

    def mean_literal_attribution(self, logit_attr, idiom_pos: tuple):
        """
        This method computes the averga DLA for the literals.

        @params
            logit_attr: DLA scores for the sentence
            idiom_pos: start and end position of the idiom
        @returns
            mean DLA of the literals
        """
        literal_tensor = t.cat((logit_attr[:idiom_pos[0]], logit_attr[idiom_pos[1]+1:]))
        if literal_tensor.size(0) == 0:
            return t.zeros(logit_attr.size(1), dtype=t.float16, device = self.device)
        else:
            return t.mean(literal_tensor, dim = 0)
    
    def compute_logit_attr(self, sent: str):
        """
        This method computes the DLA score for one sentence.

        @params
            sent: processed sentence
        @returns
            DLA scores of the idiom and the literal tokens in the sentence
        """
        if self.cache == None:
            self.get_cache(sent)
        tokens = self.model.to_tokens(sent)

        with t.inference_mode():
            if self.residual_stack == None:
                self.residual_stack = self.cache.get_full_resid_decomposition(expand_neurons=False, return_labels=False)
            logit_attr = self.logit_attrs(tokens, has_batch_dim=False)

            self.residual_stack = None
            self.cache = None
            del tokens
            t.cuda.empty_cache()

            return t.einsum("ij->ji", logit_attr)
        
    def compute_logit_attr_batched(self, batch, split_file: str):
        """
        This method computes the DLA score for a batch.

        @params
            batch: batch of sentences
            split_file: file for the intermediate results
        """
        if self.labels == None:
            self.get_labels(batch["sentence"][0])
        
        batch_split_attr = t.zeros(len(batch["sentence"]), 2, len(self.labels), device=self.device, dtype=t.float16)

        for i in range(len(batch["sentence"])):
            sent_score = self.compute_logit_attr(batch["sentence"][i])
            
            idiom_pos = batch["idiom_pos"][i]
            if batch["idiom_pos"][i][1] >= sent_score.size(0): 
                idiom_pos = [batch["idiom_pos"][i][0], sent_score.size(0)-1]
            batch_split_attr[i] = self.split_logit_attribution(sent_score, idiom_pos)

            del sent_score
            del idiom_pos
            t.cuda.empty_cache()
        
        if self.split_attr != None:
            self.split_attr = t.cat((self.split_attr, batch_split_attr), dim = 0)
        else:
            self.split_attr = batch_split_attr  
        t.save(self.split_attr, split_file) 

        del batch_split_attr
        t.cuda.empty_cache()

    def get_labels(self, sent: str):
        """
        This method extracts the component labels.
        """
        self.get_cache(sent)

        with t.inference_mode():
            self.residual_stack, self.labels = self.cache.get_full_resid_decomposition(expand_neurons=False, return_labels=True)
        print(f"\nComputing logit attribution for the following components:\n{self.labels}")
    
    def logit_attrs(self, tokens, has_batch_dim: bool = False):
        """
        This method computes the DLA score for a sentence.

        @params
            tokens: tokenized sentence
            has_batch_dim: True if the calculation is performed on a batch
        @returns
            logit_attrs: DLA scores
        """
        if isinstance(tokens, str):
            tokens = t.as_tensor(self.model.to_single_token(tokens))
        elif isinstance(tokens, int):
            tokens = t.as_tensor(tokens)

        logit_directions = self.model.tokens_to_residual_directions(tokens)

        scaled_residual_stack = self.cache.apply_ln_to_stack(
            self.residual_stack,
            layer=-1,
            pos_slice=None,
            batch_slice=None,
            has_batch_dim=has_batch_dim,
        ) 

        logit_attrs = (scaled_residual_stack * logit_directions).sum(dim=-1)
        return logit_attrs
        
    def explore_tensor(self):
        """
        Sanity check for the results.
        """
        print(f"The grouped attribution of the first sentence for {self.labels[0]} is:\n{self.split_attr[0, :, 0]}")

if __name__ == "__main__":
    loaded_tensor = t.load("./scores/logit_attribution/pythia-14m/grouped_attr_formal_0_3.pt", map_location=t.device("cpu"))
    print(f"Loaded tensor with size: {loaded_tensor.size()}")