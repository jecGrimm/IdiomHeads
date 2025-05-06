from transformer_lens import utils, HookedTransformer
from functools import partial
import torch as t
import re
from collections import defaultdict
import json

class Ablation():
    def __init__(self, model, ablation_heads = [(0, 0)]):
        self.model = model
        self.model.cfg.use_attn_result = True
        self.ablation_heads = ablation_heads
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.predictions = defaultdict(list)
        self.logit_diffs = None
        self.loss_diffs = None

    def ablate_head_batched(self, batch, ckp_file):
        batched_logit_diff = t.zeros(len(batch["sentence"]), len(self.ablation_heads), dtype=t.float16, device=self.device)
        batched_loss_diff = t.zeros(len(batch["sentence"]), len(self.ablation_heads), dtype=t.float16, device=self.device)
        for i in range(len(batch["sentence"])):
            prompt, correct_tok = self.get_correct_toks(batch["tags"][i], batch["tokenized"][i])

            if prompt != None and correct_tok != None:
                self.predictions["prompt"].append(prompt)
                self.predictions["correct_token"].append(" " + correct_tok) # needs a preceeding spaces due to the tokenization
                correct_idx = self.model.to_tokens(" " + correct_tok, prepend_bos = False).squeeze()
                
                if correct_idx.dim() != 0:
                    correct_idx = correct_idx[0]
                self.predictions["correct_index"].append(int(correct_idx))

                batched_logit_diff[i], batched_loss_diff[i] = self.ablate_head(prompt, correct_idx)

        if self.logit_diffs != None:
            self.logit_diffs = t.cat((self.logit_diffs, batched_logit_diff), dim = 0)
        else:
            self.logit_diffs = batched_logit_diff  

        if self.loss_diffs != None:
            self.loss_diffs = t.cat((self.loss_diffs, batched_loss_diff), dim = 0)
        else:
            self.loss_diffs = batched_loss_diff  
        
        del batched_logit_diff
        del batched_loss_diff
        t.cuda.empty_cache()

        t.save(self.logit_diffs, ckp_file + "_logit_ckp.pt")  
        t.save(self.loss_diffs, ckp_file + "_loss_ckp.pt")  

        with open(ckp_file + "_ckp.json", "w", encoding = "utf-8") as f:
            json.dump(self.predictions, f)  

    def ablate_head(self, prompt, correct_idx):
        orig_logits, orig_loss = self.model(prompt, return_type = "both")
        orig_logits = orig_logits.to(self.device).squeeze()
        orig_loss = orig_loss.to(self.device)

        orig_pred, orig_rank = self.get_prediction(orig_logits, correct_idx)
        self.predictions["original_prediction"].append(orig_pred)
        self.predictions["original_rank"].append(orig_rank)

        loss_diff = t.zeros(len(self.ablation_heads), dtype=t.float16, device=self.device)
        logit_diff = t.zeros(len(self.ablation_heads), dtype=t.float16, device=self.device)        
        for i in range(len(self.ablation_heads)):
            layer_to_ablate, head_index_to_ablate = self.ablation_heads[i]
            self.head_idx = head_index_to_ablate

            ablated_logits, ablated_loss = self.model.run_with_hooks(
                prompt,
                return_type="both",
                fwd_hooks=[(
                    utils.get_act_name("result", layer_to_ablate),
                    self.head_ablation_hook
                    )]
                )
            ablated_logits = ablated_logits.to(self.device).squeeze()
            ablated_loss = ablated_loss.to(self.device)
            
            logit_diff[i] = orig_logits[-1, correct_idx] - ablated_logits[-1, correct_idx]
            loss_diff[i] = ablated_loss - orig_loss 

            ablated_pred, ablated_rank = self.get_prediction(ablated_logits, correct_idx)
            self.predictions[f"L{layer_to_ablate}H{head_index_to_ablate}_prediction"].append(ablated_pred)
            self.predictions[f"L{layer_to_ablate}H{head_index_to_ablate}_rank"].append(int(ablated_rank))
            #self.predictions[f"L{layer_to_ablate}H{head_index_to_ablate}"].append(self.get_prediction(ablated_logits))

            del ablated_logits
            del ablated_loss
            del ablated_pred
            del ablated_rank
            t.cuda.empty_cache()
        
        del orig_logits
        del orig_loss
        del orig_pred
        del orig_rank
        t.cuda.empty_cache()
        
        return logit_diff, loss_diff

    def get_correct_toks(self, tags, toks):
        if len(toks) > 1:
            correct_id = max([i for i in range(len(tags)) if "IDIOM" in tags[i]])

            if correct_id >= len(toks):
                correct_id = len(toks)-1
            return self.remove_spaces([" ".join(toks[:correct_id])])[0], toks[correct_id]
        else:
            return None, None
        
    def remove_spaces(self, sent_list: list):
        '''
        This method transforms the tokenized sentences into normal sentences.

        @param sent_list: tokenized sentences 
        @returns cleaned_sents: list of the normal sentences
        '''
        cleaned_sents = []
        for sent in sent_list:
            space_matches = set(re.findall(r"(‘ | ['’\.,?!]| $)", sent))
            del_space_matches = {space_match:space_match.replace(' ', '') for space_match in space_matches}

            for space_match, del_space in del_space_matches.items():
                sent = sent.replace(space_match, del_space)
            cleaned_sents.append(sent)
        return cleaned_sents
        

    # We define a head ablation hook
    # The type annotations are NOT necessary, they're just a useful guide to the reader
    #
    def head_ablation_hook(
        self,
        value,#: Float[torch.Tensor, "batch pos head_index d_head"]
        hook, #: HookPoint
    ):# -> Float[torch.Tensor, "batch pos head_index d_head"]:
        #print(f"Shape of the value tensor: {value.shape}")
        value[:, :, self.head_idx, :] = 0.0
        return value
    
    def get_prediction(self, logits, correct_idx):
        sorted_probs = logits.softmax(dim=-1)[-1].argsort(descending = True)
        top_pred = self.model.to_string(sorted_probs[0]) 
        rank = int(t.where(correct_idx == sorted_probs)[0])
        return top_pred, rank
    
    def get_ablation_function(ablate_to_mean, head_to_ablate, component="HEAD"):
        # @deprecated
        def head_ablation_hook(
            value, #: TT["batch", "pos", "head_index", "d_head"],  # noqa: F821
            hook, #: HookPoint,
        ): # -> TT["batch", "pos", "head_index", "d_head"]:  # noqa: F821
            print(f"Shape of the value tensor: {value.shape}")

            if ablate_to_mean:
                value[:, :, head_to_ablate, :] = value[
                    :, :, head_to_ablate, :
                ].mean(dim=-1, keepdim=True)
            else:
                value[:, :, head_to_ablate, :] = 0.0
            return value
        
        if component == "HEAD":
            return head_ablation_hook
    
    def explore_tensor(self):
        print(f"The logit diff of the first sentence for the first ablated head is: {self.logit_diffs[0][0]}")
        print(f"The loss diff of the first sentence for the first ablated head is: {self.loss_diffs[0][0]}")
        
        if self.predictions["prompt"] != []:
            print(f"\nThe predictions for the first sentence are:\nPrompt: {self.predictions["prompt"][0]}\nCorrect Token: {self.predictions["correct_token"][0]}\nOriginal Prediction: {self.predictions["original_prediction"][0]}\nFirst Ablated Prediction Rank: {self.predictions[f"L{self.ablation_heads[0][0]}H{self.ablation_heads[0][1]}_rank"][0]}")
        else:
            print("\nNo predictions available!")

if __name__ == "__main__":
    model: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-14m", dtype="bfloat16") # bfloat 16, weil float 16 manchmal auf der CPU nicht geht
    model.cfg.use_attn_result = True
    logits, cache = model.run_with_cache("Test this")
    print(cache.keys())

