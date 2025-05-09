from transformer_lens import utils, HookedTransformer
from functools import partial
import torch as t
import re
from collections import defaultdict
import json

class Ablation():
    def __init__(self, model, ablation_heads = [(0, 0)], start = 0):
        self.model = model
        self.model.cfg.use_attn_result = True
        self.ablation_heads = [[comp] for comp in set([head for head_group in ablation_heads for head in head_group])]
        self.ablation_heads += ablation_heads
        for i in range(3):
            top_group = []
            for group in ablation_heads:
                top_group.append(group[i])    
            self.ablation_heads.append(top_group)

        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.predictions = defaultdict(list)
        self.orig_logits = None
        self.orig_loss = None

        self.logit_diffs = None
        self.loss_diffs = None

        self.sent_idx = start
    
    def load_predictions(self, prediction_path):
        orig_pred_file = prediction_path + "predictions_original.json"
        orig_loss_file = prediction_path + "loss_original.pt"

        try:
            with open(orig_pred_file, 'r', encoding="utf-8") as f:
                self.predictions = json.load(f)

            self.orig_loss = t.load(orig_loss_file, map_location=t.device(self.device))
        except:
            return None
        
    
    def create_original_predictions(self, batch):
        batched_orig_loss = t.zeros(len(batch["sentence"]), dtype=t.float16, device=self.device)
        add_loss = True
        for i in range(len(batch["sentence"])):
            loss = self.clean_run(batch["tags"][i], batch["tokenized"][i])
            if loss != None:
                batched_orig_loss[i] = loss
            else:
                if len(batch["sentence"]) == 1:
                    add_loss = False
        
        if add_loss:
            if self.orig_loss != None:
                self.orig_loss = t.cat((self.orig_loss, batched_orig_loss), dim = 0)
            else:
                self.orig_loss = batched_orig_loss  
        
        del batched_orig_loss
        t.cuda.empty_cache()

    def clean_run(self, tags, tokenized):
        prompt, correct_tok = self.get_correct_toks(tags, tokenized)

        if prompt != None and correct_tok != None:
            self.predictions["prompt"].append(prompt)
            self.predictions["correct_token"].append(" " + correct_tok) # needs a preceeding spaces due to the tokenization
            correct_idx = self.model.to_tokens(" " + correct_tok, prepend_bos = False).squeeze()
            
            if correct_idx.dim() != 0:
                correct_idx = correct_idx[0]
            self.predictions["correct_index"].append(int(correct_idx))

            orig_loss = self.model(prompt, return_type = "loss")
            orig_loss = orig_loss.to(self.device)

            return orig_loss
        else:
            return None
    
    def ablate_head_batched(self, batch, ckp_file):
        batched_logit_diff = t.zeros(len(batch["sentence"]), len(self.ablation_heads), dtype=t.float16, device=self.device)
        batched_loss_diff = t.zeros(len(batch["sentence"]), len(self.ablation_heads), dtype=t.float16, device=self.device)
        for i in range(len(batch["sentence"])):
            batched_logit_diff[i], batched_loss_diff[i] = self.ablate_head(self.predictions["prompt"][self.sent_idx], self.predictions["correct_index"][self.sent_idx])
            self.sent_idx += 1

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
        # clean run
        clean_logit = self.model(prompt, return_type = "logits").to(self.device).squeeze()
        clean_pred, clean_rank = self.get_prediction(clean_logit, correct_idx)
        self.predictions["original_prediction"].append(clean_pred)
        self.predictions["original_rank"].append(clean_rank)
        clean_loss = self.orig_loss[self.sent_idx]

        loss_diff = t.zeros(len(self.ablation_heads), dtype=t.float16, device=self.device)
        logit_diff = t.zeros(len(self.ablation_heads), dtype=t.float16, device=self.device)        
        for i in range(len(self.ablation_heads)):
            abl_comp = self.ablation_heads[i]
            name = ""
            fwd_hook = []
            if len(abl_comp) == 1:
                layer_to_ablate, head_index_to_ablate = abl_comp[0]
                hook_fn = partial(self.head_ablation_hook, head_index_to_ablate = head_index_to_ablate)
                fwd_hook = [(utils.get_act_name("result", layer_to_ablate), hook_fn)]
                name = f"L{layer_to_ablate}H{head_index_to_ablate}"
            else:
                for comp in abl_comp:
                    layer_to_ablate, head_index_to_ablate = comp
                    hook_fn = partial(self.head_ablation_hook, head_index_to_ablate = head_index_to_ablate)
                    fwd_hook.append((utils.get_act_name("result", layer_to_ablate), hook_fn))
                    name += f"_L{layer_to_ablate}H{head_index_to_ablate}"
                name = name[1:]

            ablated_logits, ablated_loss = self.model.run_with_hooks(
                prompt,
                return_type="both",
                fwd_hooks = fwd_hook
                )
            
            ablated_logits = ablated_logits.to(self.device).squeeze()
            ablated_loss = ablated_loss.to(self.device)
            
            logit_diff[i] = clean_logit[-1, correct_idx] - ablated_logits[-1, correct_idx]
            loss_diff[i] = ablated_loss - clean_loss 

            ablated_pred, ablated_rank = self.get_prediction(ablated_logits, correct_idx)
            self.predictions[f"{name}_prediction"].append(ablated_pred)
            self.predictions[f"{name}_rank"].append(int(ablated_rank))
            #self.predictions[f"L{layer_to_ablate}H{head_index_to_ablate}"].append(self.get_prediction(ablated_logits))

            del ablated_logits
            del ablated_loss
            del ablated_pred
            del ablated_rank
            t.cuda.empty_cache()
        
        del clean_logit
        del clean_loss
        del clean_pred
        del clean_rank
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
        head_index_to_ablate
    ):# -> Float[torch.Tensor, "batch pos head_index d_head"]:
        #print(f"Shape of the value tensor: {value.shape}")
        value[:, :, head_index_to_ablate, :] = 0.0
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
            print(f"\nThe predictions for the first sentence are:\nPrompt: {self.predictions["prompt"][0]}\nCorrect Token: {self.predictions["correct_token"][0]}\nOriginal Prediction: {self.predictions["original_prediction"][0]}\nFirst Ablated Prediction Rank: {self.predictions[f"L{self.ablation_heads[0][0][0]}H{self.ablation_heads[0][0][1]}_rank"][0]}")
        else:
            print("\nNo predictions available!")

if __name__ == "__main__":
    model: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-14m", dtype="bfloat16") # bfloat 16, weil float 16 manchmal auf der CPU nicht geht
    model.cfg.use_attn_result = True
    logits, cache = model.run_with_cache("Test this")
    print(cache.keys())

