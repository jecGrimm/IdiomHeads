from cli import CLI
import os
from transformer_lens import HookedTransformer
from data import EPIE_Data
from logit_attribution import LogitAttribution
import torch as t

cli = CLI() # parses arguments
os.makedirs(f"./scores/logit_attribution/{cli.model_name}", exist_ok=True)

model: HookedTransformer = HookedTransformer.from_pretrained(cli.full_model_name) # cannot load bfloat16 because logit_attr does not work with that dtype

model.eval()
epie = EPIE_Data()
scorer = LogitAttribution(model)
print(f"Running on device {scorer.device}.")

for i in range(len(cli.data_split)):
    split = cli.data_split[i]
    if split == "formal":
        data = epie.create_hf_dataset(epie.formal_sents[cli.start:cli.end], epie.tokenized_formal_sents[cli.start:cli.end], epie.tags_formal[cli.start:cli.end])
    elif split == "trans":
        data = epie.create_hf_dataset(epie.trans_formal_sents[cli.start:cli.end], epie.tokenized_trans_formal_sents[cli.start:cli.end], epie.tags_formal[cli.start:cli.end])
    else:
        raise Exception(f"Split {split} not in the dataset, please choose either formal or trans as optional argument -d")
    data = data.add_column("idiom_pos", scorer.idiom_positions[cli.start:cli.end])

    if cli.batch_sizes[i] == None:
        batch_size = 1
    else:
        batch_size = int(cli.batch_sizes[i])
    
    grouped_file = f"./scores/logit_attribution/{cli.model_name}/grouped_attr_{split}_{cli.start}_{cli.end}.pt"
    
    data.map(lambda batch: scorer.compute_logit_attr_batched(batch, grouped_file), batched=True, batch_size = batch_size)

    scorer.explore_tensor()
    
    scorer.split_attr = None
    t.cuda.empty_cache()
