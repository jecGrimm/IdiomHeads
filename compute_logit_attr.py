from cli import CLI
import os
from transformer_lens import HookedTransformer
from data import EPIE_Data
from logit_attribution import LogitAttribution
import torch as t

cli = CLI() # parses arguments
os.makedirs(f"./scores/logit_attribution/{cli.model_name}", exist_ok=True)
os.makedirs(f"./plots/{cli.model_name}", exist_ok=True)
t.set_grad_enabled(False)

model: HookedTransformer = HookedTransformer.from_pretrained(cli.full_model_name, dtype="bfloat16") # cannot load bfloat16 because logit_attr does not work with that dtype

model.eval()
epie = EPIE_Data()
scorer = LogitAttribution(model, filename = cli.idiom_file)
print(f"Running on device {scorer.device}.")

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
    else:
        raise Exception(f"Split {split} not in the dataset, please choose either formal or trans as optional argument -d")
    
    if scorer.idiom_positions == []:
        data.map(lambda batch: scorer.get_all_idiom_pos(batch), batched=True, batch_size=batch_size)
        scorer.store_all_idiom_pos(cli.idiom_file)
    data = data.add_column("idiom_pos", scorer.idiom_positions[start:end])
    
    grouped_file = f"./scores/logit_attribution/{cli.model_name}/grouped_attr_{split}_{start}_{end}.pt"
    
    data.map(lambda batch: scorer.compute_logit_attr_batched(batch, grouped_file), batched=True, batch_size = batch_size)

    scorer.explore_tensor()

    # PLOT
    #plot_logit_attribution_split(scorer.split_attr, title = "Mean logit attribution: Idiom vs. Literal", x_labels = scorer.labels, filename = f"./plots/{cli.model_name}/heat_{split}_logit_attr.png")
    
    scorer.split_attr = None
    t.cuda.empty_cache()
