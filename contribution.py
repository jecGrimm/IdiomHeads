import sys
sys.path.append("llm-transparency-tool")

from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
from llm_transparency_tool.routes.contributions import get_attention_contributions

import torch as t
import json
import os
from merge_tokenizers import PythonGreedyCoverageAligner, types

class Contribution:
    def __init__(self, model, filename: str = "llama_formal_idiom_pos_contribution.json"):
        """
        This class computes the contribution of a head.

        @params
            model: examined model
            filename: file with the idiom positions
        """
        self.model = model
        self.n_layers = self.model._model.cfg.n_layers
        self.n_heads = self.model._model.cfg.n_heads
        self.total_contribution = None
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.idiom_positions = self.load_all_idiom_pos(filename)
        self.labels = None
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
            model_str_tokens= self.model._model.to_str_tokens(sent)
            aligned_positions = self.align_tokens(sent, batch["tokenized"][i], model_str_tokens)
            self.idiom_positions.append(self.get_idiom_pos(aligned_positions, batch["tags"][i]))

    def align_tokens(self, sent: str, tokenized_sent: list, model_str_tokens: list):
        """
        This method aligns the tokens retrieved by EPIE with the tokens retrieved by the model.

        @params
            sent: natural sentence
            tokenized_sent: sentence tokenized by EPIE
            model_str_tokens: sentence tokenized by the model
        @returns
            list of the aligned token positions
        """
        aligned = self.aligner.align(
            types.TokenizedSet(tokens=[tokenized_sent, model_str_tokens], text=sent)
        )
        return list(aligned[0])
    
    def get_idiom_pos(self, aligned_positions, tags: list):
        """
        This method extracts the position of the idioms in the sentence tokenized by the model.

        @params
            aligned_positions: aligned EPIE and model positions
            tags: idiom labels
        @returns 
            start and end position of the idiom
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

    def compute_contribution_batched(self, batch, split_file: str):
        """
        This method computes the contribution for a batch of sentences.

        @params
            batch: batch of sentences
            split_file: output file
        """
        batch_split_contr = t.zeros(len(batch["sentence"]), self.n_layers, 2, self.n_heads, device=self.device, dtype=t.float16)

        for i in range(len(batch["sentence"])):
            batch_split_contr[i] = self.compute_contribution(batch["sentence"][i], batch["idiom_pos"][i])
        
        if self.total_contribution != None:
            self.total_contribution = t.cat((self.total_contribution, batch_split_contr), dim = 0)
        else:
            self.total_contribution = batch_split_contr
        t.save(self.total_contribution, split_file) 

        del batch_split_contr
        t.cuda.empty_cache()
      
    def compute_contribution(self, sentence: str, idiom_pos: tuple):
        """
        This method computes the contribution of an attention head for a single sentence.

        @params
            sentence: processed sentence
            idiom_pos: start and end position of the idiom in the sentence
        @returns 
            contributions: contribution of the head for literal and idiom tokens
        """
        self.model.run([sentence])
        
        contributions = t.zeros(self.n_layers, 2, self.n_heads, device=self.device, dtype=t.float16)
        for layer in range(self.n_layers):
            resid_pre = self.model.residual_in(layer)[0].unsqueeze(0)
            resid_mid = self.model.residual_after_attn(layer)[0].unsqueeze(0)
            decomposed_attn = self.model.decomposed_attn(0, layer).unsqueeze(0)

            head_contrib, _ = get_attention_contributions(resid_pre, resid_mid, decomposed_attn)

            del resid_pre
            del resid_mid
            del decomposed_attn
            t.cuda.empty_cache()

            if idiom_pos[1] >= head_contrib.size(0): 
                idiom_pos = [idiom_pos[0], head_contrib.size(0)-1]

            layer_contrib = head_contrib[0, idiom_pos[1], :, :] 

            del head_contrib
            t.cuda.empty_cache()

            contributions[layer] = self.split_layer_contribution(layer_contrib, idiom_pos)
        
        return contributions

    def split_layer_contribution(self, layer_contribution, idiom_pos: tuple):
        """
        This method splits the contribution up into idiom and literal tokens.

        @params
            layer_contribution: contribution for the full sentence
            idiom_pos: start and end positions of the idiom
        @return
            stack of the grouped idiom and literal contribution
        """
        idiom_contr = self.mean_idiom_contribution(layer_contribution, idiom_pos)
        literal_contr = self.mean_literal_contribution(layer_contribution, idiom_pos)
        return t.vstack((idiom_contr, literal_contr))

    def mean_idiom_contribution(self, layer_contribution, idiom_pos: tuple):
        """
        This method computes the mean over contribution of the idiom tokens.

        @params
            layer_contribution: contribution of the full sentence
            idiom_pos: start and end positions of the idiom
        @returns
            averaged contribution of the idiom tokens
        """
        idiom_tensor = layer_contribution[idiom_pos[0]:idiom_pos[1]+1]
        if idiom_tensor.size(0) == 0:
            return t.zeros(layer_contribution.size(1), dtype=t.float16, device = self.device)
        else:
            return t.mean(idiom_tensor, dim = 0)

    def mean_literal_contribution(self, layer_contribution, idiom_pos: tuple):
        """
        This method computes the mean over contribution of the literal tokens.

        @params
            layer_contribution: contribution of the full sentence
            idiom_pos: start and end positions of the idiom
        @returns
            averaged contribution of the literal tokens
        """
        literal_tensor = t.cat((layer_contribution[:idiom_pos[0]], layer_contribution[idiom_pos[1]+1:]))
        if literal_tensor.size(0) == 0:
            return t.zeros(layer_contribution.size(1), dtype=t.float16, device = self.device)
        else:
            return t.mean(literal_tensor, dim = 0)

    def explore_tensor(self):
        """
        Sanity check for the results of the experiment.
        """
        print(f"The grouped contribution of the first sentence for L0H0 is:\n{self.total_contribution[0, 0, :, 0]}")


if __name__ == "__main__":
  model = TransformerLensTransparentLlm("meta-llama/Llama-3.2-1B-Instruct", dtype = t.bfloat16)