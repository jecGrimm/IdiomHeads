from transformer_lens import (
    HookedTransformer,
)
from data import EPIE_Data
from idiom_score import Scorer
import os
import argparse

parser = argparse.ArgumentParser(prog='idiom head detector')
parser.add_argument('--model_name', '-m', help='model to run the experiment with', default="EleutherAI/pythia-1.4b")

model_name = parser.parse_args().model_name

#model: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-14m")
#pythia_1_4b: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-1.4b")
model: HookedTransformer = HookedTransformer.from_pretrained(model_name)

epie = EPIE_Data()
scorer = Scorer(model)
print(f"Running on device: {scorer.device}")

trans_data = epie.create_hf_dataset(epie.trans_sents, epie.tokenized_trans_sents, epie.tags_formal)
trans_scores = scorer.create_data_score_tensor(trans_data)

scorer.explore_tensor(trans_scores)

if not os.path.isdir("./scores"):
    os.mkdir("./scores")
scorer.save_tensor(trans_scores, f"./scores/{model_name.split('/')}_trans.pt")