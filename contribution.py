import sys
sys.path.append("llm-transparency-tool")

from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
from llm_transparency_tool.routes.contributions import get_attention_contributions

import torch as t
import argparse
from transformers import HfArgumentParser
import json
import os
from merge_tokenizers import PythonGreedyCoverageAligner, types

class Contribution:
    def __init__(self, model, filename = "llama_formal_idiom_pos.json"):
        self.model = model
        self.n_layers = self.model._model.cfg.n_layers
        self.n_heads = self.model._model.cfg.n_heads
        self.total_contribution = None
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.idiom_positions = self.load_all_idiom_pos(filename)
        self.labels = None
        self.aligner = PythonGreedyCoverageAligner()

    def load_all_idiom_pos(self, filename):
        if os.path.isfile(filename):
            with open(filename, 'r', encoding = "utf-8") as f:
                return json.load(f) 
        else:
            return [] 
        
    def get_all_idiom_pos(self, batch):
        for i in range(len(batch["sentence"])):
            sent = batch["sentence"][i]
            model_str_tokens= self.model._model.to_str_tokens(sent)
            aligned_positions = self.align_tokens(sent, batch["tokenized"][i], model_str_tokens)
            self.idiom_positions.append(self.get_idiom_pos(aligned_positions, batch["tags"][i]))

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

    def store_all_idiom_pos(self, filename):
        with open(filename, 'w', encoding = "utf-8") as f:
            json.dump(self.idiom_positions, f)

    def compute_contribution_batched(self, batch, split_file):
        batch_split_contr = t.zeros(len(batch["sentence"]), self.n_layers, 2, self.n_heads, device=self.device, dtype=t.float16)

        for i in range(len(batch["sentence"])):
            #idiom_pos = batch["idiom_pos"][i]
            # if batch["idiom_pos"][i][1] >= sent_score.size(0): 
            #     idiom_pos = [batch["idiom_pos"][i][0], sent_score.size(0)-1]
            
            batch_split_contr[i] = self.compute_contribution(batch["sentence"][i], batch["idiom_pos"][i])

            # del sent_score
            # del idiom_pos
            # t.cuda.empty_cache()
        
        if self.total_contribution != None:
            self.total_contribution = t.cat((self.total_contribution, batch_split_contr), dim = 0)
        else:
            self.total_contribution = batch_split_contr
        t.save(self.total_contribution, split_file) 

        del batch_split_contr
        t.cuda.empty_cache()
      
    def compute_contribution(self, sentence, idiom_pos):
      self.model.run([sentence])
      
      contributions = t.zeros(self.n_layers, 2, self.n_heads, device=self.device, dtype=t.float16)
      # # VON LEA
      for layer in range(self.n_layers):
        resid_pre = self.model.residual_in(layer)[0].unsqueeze(0)
        #print("resid_pre:", resid_pre.size())
        resid_mid = self.model.residual_after_attn(layer)[0].unsqueeze(0)
        #print("resid_mid:", resid_mid.size())
        decomposed_attn = self.model.decomposed_attn(0, layer).unsqueeze(0)
        #print("decomposed_attn:", decomposed_attn.size())

        head_contrib, _ = get_attention_contributions(resid_pre, resid_mid, decomposed_attn)
        #print("head_contrib:", head_contrib.size())

        del resid_pre
        del resid_mid
        del decomposed_attn
        t.cuda.empty_cache()

        if idiom_pos[1] >= head_contrib.size(0): 
            idiom_pos = [idiom_pos[0], head_contrib.size(0)-1]

        # # [batch pos key_pos head] -> [key_pos head]
        layer_contrib = head_contrib[0, idiom_pos[1], :, :] 
        #print("layer_contrib:", layer_contrib.size())

        del head_contrib
        t.cuda.empty_cache()

        contributions[layer] = self.split_layer_contribution(layer_contrib, idiom_pos)
      
      return contributions

    def split_layer_contribution(self, layer_contribution, idiom_pos):
        idiom_contr = self.mean_idiom_contribution(layer_contribution, idiom_pos)
        literal_contr = self.mean_literal_contribution(layer_contribution, idiom_pos)
        return t.vstack((idiom_contr, literal_contr))

    def mean_idiom_contribution(self, layer_contribution, idiom_pos):
        idiom_tensor = layer_contribution[idiom_pos[0]:idiom_pos[1]+1]
        if idiom_tensor.size(0) == 0:
            return t.zeros(layer_contribution.size(1), dtype=t.float16, device = self.device)
        else:
            return t.mean(idiom_tensor, dim = 0)

    def mean_literal_contribution(self, layer_contribution, idiom_pos):
        literal_tensor = t.cat((layer_contribution[:idiom_pos[0]], layer_contribution[idiom_pos[1]+1:]))
        if literal_tensor.size(0) == 0:
            return t.zeros(layer_contribution.size(1), dtype=t.float16, device = self.device)
        else:
            return t.mean(literal_tensor, dim = 0)

    def explore_tensor(self):
        print(f"The grouped contribution of the first sentence for L0H0 is:\n{self.total_contribution[0, 0, :, 0]}")


if __name__ == "__main__":
  model = TransformerLensTransparentLlm("meta-llama/Llama-3.2-1B-Instruct", dtype = t.bfloat16)
  print("CFG:", model._model.cfg)
  # model.set_ungroup_grouped_query_attention(True)
#   model.run(["Test this"])

#   # # VON LEA
#   layer, B0, source_token = 0, 0, 0
#   resid_pre = model.residual_in(layer)[B0].unsqueeze(0)
#   print("resid_pre:", resid_pre.size())
#   resid_mid = model.residual_after_attn(layer)[B0].unsqueeze(0)
#   print("resid_mid:", resid_mid.size())
#   decomposed_attn = model.decomposed_attn(B0, layer).unsqueeze(0)
#   print("decomposed_attn:", decomposed_attn.size())

#   head_contrib, _ = get_attention_contributions(resid_pre, resid_mid, decomposed_attn)
#   print("head_contrib:", head_contrib.size())

#   # # [batch pos key_pos head] -> [head]
#   flat_contrib = head_contrib[0, -1, source_token, :] # #TODO token 2 is the noun, do this dynamically
#   print("flat_contrib:", flat_contrib.size())