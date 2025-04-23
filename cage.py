from cli import CLI
import os
from transformer_lens import HookedTransformer
import torch as t
from data import EPIE_Data
import json

class Cage:
    def __init__(self, model):
        self.model = model
        self.model.cfg.use_attn_result = True
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.batch_num = 0

    def cage_batched(self, batch, path):
        residual = None
        for i in range(len(batch["sentence"])):
            cache = self.get_cache(batch["sentence"][i])
             
            # TODO: Bist auf Result wird nur die Datei 0 gespeichert
            t.save(cache.stack_activation("pattern"), f"{path}/pattern/{self.batch_num}.pt")
            t.save(cache.stack_activation("result"), f"{path}/result/{self.batch_num}.pt")
            
            if self.batch_num == 0:
                residual, labels = cache.get_full_resid_decomposition(expand_neurons=False, return_labels=True)

                with open(f"{path}/residual/labels.json", 'w', encoding="utf-8") as f:
                    json.dump(labels, f)   

                del labels
                t.cuda.empty_cache()
            else:
                residual = cache.get_full_resid_decomposition(expand_neurons=False, return_labels=False)
            t.save(residual, f"{path}/residual/{self.batch_num}.pt")

            del cache
            del residual
            t.cuda.empty_cache()

            self.batch_num += 1


    def get_cache(self, sent):
        idiom_tokens = self.model.to_tokens(sent)
        _, cache = self.model.run_with_cache(idiom_tokens, remove_batch_dim=True)
        return cache.to(self.device)
    
    def explore_tensor(self, path):
        pattern = t.load(f"{path}/pattern/0.pt")
        print(f"Pattern: {pattern.size()}")

        del pattern
        t.cuda.empty_cache()

        result = t.load(f"{path}/result/0.pt")
        print(f"Result: {result.size()}")

        del result
        t.cuda.empty_cache()

        residual = t.load(f"{path}/residual/0.pt")
        print(f"Residual: {residual.size()}")

        del residual
        t.cuda.empty_cache()



if __name__ == "__main__":
    cli = CLI()

    # Saves computation time
    t.set_grad_enabled(False)

    model: HookedTransformer = HookedTransformer.from_pretrained(cli.full_model_name, dtype="bfloat16")
    epie = EPIE_Data()
    cage = Cage(model)
    print(f"Running on device {cage.device}.")

    for i in range(len(cli.data_split)):
        if cli.batch_sizes[i] == None:
            batch_size = 1
        else:
            batch_size = int(cli.batch_sizes[i])

        split = cli.data_split[i]
        print("\nProcessing split: ", split)

        os.makedirs(f"./cage/{cli.model_name}/{split}/pattern", exist_ok=True)
        os.makedirs(f"./cage/{cli.model_name}/{split}/result", exist_ok=True)
        os.makedirs(f"./cage/{cli.model_name}/{split}/residual", exist_ok=True)

        if split == "formal":
            data = epie.create_hf_dataset(epie.formal_sents[cli.start:cli.end], epie.tokenized_formal_sents[cli.start:cli.end], epie.tags_formal[cli.start:cli.end])
        elif split == "trans":
            data = epie.create_hf_dataset(epie.trans_formal_sents[cli.start:cli.end], epie.tokenized_trans_formal_sents[cli.start:cli.end], epie.tags_formal[cli.start:cli.end])
        else:
            raise Exception(f"Split {split} not in the dataset, please choose either formal or trans as optional argument -d")
        
        data.map(lambda batch: cage.cage_batched(batch, f"./cage/{cli.model_name}/{split}"), batched=True, batch_size = batch_size)
        
        cage.explore_tensor(f"./cage/{cli.model_name}/{split}")

