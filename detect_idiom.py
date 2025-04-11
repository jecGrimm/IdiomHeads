from transformer_lens import (
    HookedTransformer,
)
from data import EPIE_Data
from idiom_score import IdiomScorer
import os
import argparse

parser = argparse.ArgumentParser(prog='idiom head detector')
parser.add_argument('--model_name', '-m', help='model to run the experiment with', default="EleutherAI/pythia-1.4b")

model_name = parser.parse_args().model_name

#model: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-14m")
#pythia_1_4b: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-1.4b")
model: HookedTransformer = HookedTransformer.from_pretrained(model_name)

epie = EPIE_Data()
scorer = IdiomScorer(model)
print(f"Running on device: {scorer.device}")

formal_data = epie.create_hf_dataset(epie.formal_sents, epie.tokenized_formal_sents, epie.tags_formal)
formal_scores = scorer.create_data_score_tensor(formal_data)

scorer.explore_tensor(formal_scores)

if not os.path.isdir("./scores"):
    os.mkdir("./scores")
scorer.save_tensor(formal_scores, f"./scores/{model_name.split('/')}_formal.pt")