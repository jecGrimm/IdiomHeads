from transformer_lens import (
    HookedTransformer,
)
from data import EPIE_Data
from ablation import Ablation
import os
import torch as t
from cli import CLI
import json
from collections import defaultdict

cli = CLI()
os.makedirs(f"./scores/ablation/{cli.model_name}", exist_ok=True)
# Saves computation time
t.set_grad_enabled(False)

model: HookedTransformer = HookedTransformer.from_pretrained(cli.model_name, dtype="bfloat16") # bfloat 16, weil float 16 manchmal auf der CPU nicht geht

model.eval()
epie = EPIE_Data()

ablation_heads = {
    "pythia-14m": [(0,0), (5,3)],
    "pythia-1.4b": [(2, 15), (16, 10), (19, 14), (15, 13)] # top heads identified by idiom score and dla
}
scorer = Ablation(model, ablation_heads=ablation_heads[cli.model_name])
print(f"Running compute_ablation on device {scorer.device}.")

for i in range(len(cli.data_split)):
    if cli.batch_sizes[i] == None:
        batch_size = 1
    else:
        batch_size = int(cli.batch_sizes[i])

    split = cli.data_split[i]
    print(f"\nProcessing split {split}:")

    start = cli.start[i]
    end = cli.end[i]

    if split == "formal":
        data = epie.create_hf_dataset(epie.formal_sents[start:end], epie.tokenized_formal_sents[start:end], epie.tags_formal[start:end])
    elif split == "trans":
        data = epie.create_hf_dataset(epie.trans_formal_sents[start:end], epie.tokenized_trans_formal_sents[start:end], epie.tags_formal[start:end])
    else:
        raise Exception(f"Split {split} not in the dataset, please choose either formal or trans as optional argument -d")

    ckp_file = f"./scores/ablation/{cli.model_name}/ablation_{split}_{start}_{end}"
    
    data.map(lambda batch: scorer.ablate_head_batched(batch, ckp_file), batched=True, batch_size = batch_size)

    scorer.explore_tensor()

    t.save(scorer.logit_diffs, f"{ckp_file}_logit.pt")
    t.save(scorer.loss_diffs, f"{ckp_file}_loss.pt")
    with open(f"{ckp_file}.json", 'w', encoding = "utf-8") as f:
        json.dump(scorer.predictions, f)

    scorer.logit_diffs = None
    scorer.loss_diffs = None
    scorer.predictions = defaultdict(list)
    t.cuda.empty_cache()
    