import torch as t
import os
import json

class LogitAttribution:
    def __init__(self, model, split = "formal"):
        self.model = model
        self.model.cfg.use_attn_result = True
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.token_attr = None
        self.split_attr = None
        self.idiom_positions = self.load_all_idiom_pos(split)
        self.labels = None

    def load_all_idiom_pos(self, split):
        if os.path.isfile(f"{split}_idiom_pos.json"):
            with open(f"{split}_idiom_pos.json", 'r', encoding = "utf-8") as f:
                return json.load(f) 
        else:
            return [] 

    def get_cache(self, sent):
        _, cache = self.model.run_with_cache(sent, remove_batch_dim=True)
        return cache.to(self.device)

    def split_logit_attribution(self, logit_attr, idiom_pos):
        idiom_attr = self.mean_idiom_attribution(logit_attr, idiom_pos)
        literal_attr = self.mean_literal_attribution(logit_attr, idiom_pos)
        return t.vstack((idiom_attr, literal_attr))

    def mean_idiom_attribution(self, logit_attr, idiom_pos):
        return t.mean(logit_attr[idiom_pos[0]:idiom_pos[1]], dim = 0)

    def mean_literal_attribution(self, logit_attr, idiom_pos):
        return t.mean(t.cat((logit_attr[:idiom_pos[0]], logit_attr[idiom_pos[1]+1:])), dim = 0)
    
    def compute_logit_attr(self, sent):
        cache = self.get_cache(sent)
        tokens = self.model.to_tokens(sent)

        with t.inference_mode():
            residual_stack = cache.get_full_resid_decomposition(expand_neurons=False, return_labels=False)
            logit_attr = cache.logit_attrs(residual_stack, tokens, has_batch_dim=False)

            del residual_stack
            del cache
            t.cuda.empty_cache()

            return t.einsum("ij->ji", logit_attr)
        
    def compute_logit_attr_batched(self, batch, split_file):
        if self.labels == None:
            self.labels = self.get_labels(batch["sentence"][0])
        
        batch_split_attr = t.zeros(len(batch["sentence"]), 2, len(self.labels))

        for i in range(len(batch["sentence"])):
            sent_score = self.compute_logit_attr(batch["sentence"][i])
            
            batch_split_attr[i] = self.split_logit_attribution(sent_score, batch["idiom_pos"][i])
        
        if self.split_attr != None:
            self.split_attr = t.cat((self.split_attr, batch_split_attr), dim = 0)
        else:
            self.split_attr = batch_split_attr  
        t.save(self.split_attr, split_file) 

        del batch_split_attr
        t.cuda.empty_cache()

    def get_labels(self, sent):
        cache = self.get_cache(sent)

        with t.inference_mode():
            _, lbls = cache.get_full_resid_decomposition(expand_neurons=False, return_labels=True)
        print("\nComputing logit attribution for the following components:\n{lbls}")
        return lbls
        
    def explore_tensor(self):
        print(f"The grouped attribution of the first sentence for {self.labels[0]} is:\n{self.split_attr[0, :, 0]}")

if __name__ == "__main__":
    loaded_tensor = t.load("./scores/logit_attribution/pythia-14m/grouped_attr_formal_0_3.pt", map_location=t.device("cpu"))
    print(f"Loaded tensor with size: {loaded_tensor.size()}")