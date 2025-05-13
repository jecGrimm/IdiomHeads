from cli import CLI
import os
from transformer_lens import HookedTransformer
from data import EPIE_Data
from contribution import Contribution
import torch as t
import sys
sys.path.append("llm-transparency-tool")

from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm

cli = CLI() # parses arguments
os.makedirs(f"./scores/contribution/{cli.model_name}", exist_ok=True)
t.set_grad_enabled(False)

model = TransformerLensTransparentLlm("meta-llama/Llama-3.2-1B-Instruct", dtype = t.bfloat16) # cannot load bfloat16 because logit_attr does not work with that dtype

model.eval()
epie = EPIE_Data()
scorer = Contribution(model)
print(f"Running compute_contribution on device {scorer.device}.")

for i in range(len(cli.data_split)):
    if cli.batch_sizes[i] == None:
        batch_size = 1
    else:
        batch_size = int(cli.batch_sizes[i])
    
    start = cli.start[i]
    end = cli.end[i]

    split = cli.data_split[i]
    print("\nProcessing split: ", split)
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
    
    grouped_file = f"./scores/contribution/{cli.model_name}/grouped_contr_{split}_{start}_{end}.pt"
    
    data.map(lambda batch: scorer.compute_contribution_batched(batch, grouped_file), batched=True, batch_size = batch_size)

    scorer.explore_tensor()

    scorer.total_contribution = None
    t.cuda.empty_cache()
