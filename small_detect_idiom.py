from transformer_lens import (
    HookedTransformer,
)
from data import EPIE_Data
from idiom_score import Scorer
import os
import torch as t

model: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-14m")
epie = EPIE_Data()
scorer = Scorer(model)

formal_data = epie.create_hf_dataset(epie.formal_sents[:3], epie.tokenized_formal_sents[:3], epie.tags_formal[:3])
formal_scores = scorer.create_data_score_tensor(formal_data)

if not os.path.isdir("./scores"):
    os.mkdir("./scores")
scorer.save_tensor(formal_scores, "./scores/test_formal.pt")

