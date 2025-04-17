from transformer_lens import (
    HookedTransformer,
)
from data import EPIE_Data
from idiom_score import IdiomScorer
import os
import torch as t
import argparse

parser = argparse.ArgumentParser(prog='idiom head detector')
parser.add_argument('--model_name', '-m', help='model to run the experiment with', default="EleutherAI/pythia-1.4b")
parser.add_argument('--data', '-d', help='list of data split that should be processed', nargs='*', default=["formal"], type=str)
parser.add_argument('--start', '-s', help='start index (inclusive)', default = 0, type=int)
parser.add_argument('--end', '-e', help='end index (exclusive)', default = None)
parser.add_argument('--batch_size', '-b', help='batch size', nargs='*', default = [None])

model_name = parser.parse_args().model_name
data_split = parser.parse_args().data
start = parser.parse_args().start

end = parser.parse_args().end
if end:
    end = int(end)

batch_sizes = parser.parse_args().batch_size
while len(batch_sizes) < len(data_split):
    batch_sizes.append(batch_sizes[-1])

if not os.path.isdir("./scores"):
    os.mkdir("./scores")

if not os.path.isdir("./scores/idiom_components"):
    os.mkdir("./scores/idiom_components")

if not os.path.isdir(f"./scores/idiom_components/{model_name.split('/')[-1]}"):
    os.mkdir(f"./scores/idiom_components/{model_name.split('/')[-1]}")

model: HookedTransformer = HookedTransformer.from_pretrained(model_name, dtype="bfloat16") # bfloat 16, weil float 16 manchmal auf der CPU nicht geht

model.eval()
epie = EPIE_Data()
scorer = IdiomScorer(model)
print(f"Running on device {scorer.device}.")

for i in range(len(data_split)):
    split = data_split[i]
    if split == "formal":
        data = epie.create_hf_dataset(epie.formal_sents[start:end], epie.tokenized_formal_sents[start:end], epie.tags_formal[start:end])
    elif split == "trans":
        data = epie.create_hf_dataset(epie.trans_formal_sents[start:end], epie.tokenized_trans_formal_sents[start:end], epie.tags_formal[start:end])
    else:
        raise Exception(f"Split {split} not in the dataset, please choose either formal or trans as optional argument -d")
    data = data.add_column("idiom_pos", scorer.idiom_positions[start:end])

    if batch_sizes[i] == None:
        batch_size = 1
    else:
        batch_size = int(batch_sizes[i])
    
    comp_file = f"./scores/idiom_components/{model_name.split('/')[-1]}/idiom_only_{split}_{start}_{end}_comp.pt"
    
    data.map(lambda batch: scorer.create_idiom_score_tensor(batch, comp_file), batched=True, batch_size = batch_size)

    scorer.explore_tensor()
    # print(scorer.components[0])
    # print(scorer.components[0].size())

    t.save(scorer.scores, f"./scores/idiom_scores/{model_name.split('/')[-1]}/idiom_only_{split}_{start}_{end}.pt")
    
    scorer.scores = None
    scorer.components = None
    t.cuda.empty_cache()