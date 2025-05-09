from transformer_lens import (
    HookedTransformer,
)
from data import EPIE_Data
from literal_score import LiteralScorer
import os
import torch as t
from cli import CLI

cli = CLI()
os.makedirs(f"./scores/literal_components/{cli.model_name}", exist_ok=True)
# Saves computation time
t.set_grad_enabled(False)

model: HookedTransformer = HookedTransformer.from_pretrained(cli.full_model_name, dtype="bfloat16") # bfloat 16, weil float 16 manchmal auf der CPU nicht geht

model.eval()
epie = EPIE_Data()
scorer = LiteralScorer(model, filename = cli.idiom_file)
print(f"Running compute_literal_only on device {scorer.device}.")

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
    elif split == "static":
        data = epie.create_hf_dataset(epie.static_sents[start:end], epie.tokenized_static_sents[start:end], epie.tags_static[start:end])
    else:
        raise Exception(f"Split {split} not in the dataset, please choose either formal or trans as optional argument -d")
    
    if scorer.idiom_positions == []:
        data.map(lambda batch: scorer.get_all_idiom_pos(batch), batched=True, batch_size=batch_size)
        scorer.store_all_idiom_pos(cli.idiom_file)
    data = data.add_column("idiom_pos", scorer.idiom_positions[start:end])

    comp_file = f"./scores/literal_components/{cli.model_name}/literal_only_{split}_{start}_{end}_comp.pt"
    
    data.map(lambda batch: scorer.create_data_score_tensor(batch, comp_file), batched=True, batch_size = batch_size)
    #data.map(lambda batch: scorer.create_idiom_score_tensor(batch, ckp_file), batched=True, batch_size = batch_size)

    scorer.explore_tensor()

    t.save(scorer.scores, f"./scores/literal_scores/{cli.model_name}/literal_only_{split}_{start}_{end}.pt")

    scorer.scores = None
    scorer.components = None
    t.cuda.empty_cache()
    