from transformer_lens import (
    HookedTransformer,
)
from data import EPIE_Data
from idiom_score import Scorer
from plot import plot_score_line, plot_score_hist, create_df
import os
import argparse

parser = argparse.ArgumentParser(prog='idiom head detector')
parser.add_argument('--model_name', '-m', help='model to run the experiment with', default="EleutherAI/pythia-14m")

model_name = parser.parse_args().model_name

#pythia_1_4b: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-1.4b")
model: HookedTransformer = HookedTransformer.from_pretrained(model_name)

# ATTENTION HEADS
# All sentences
epie = EPIE_Data()
all_idiom_sents = epie.formal_sents + epie.static_sents
all_tokenized_sents = epie.tokenized_formal_sents + epie.tokenized_static_sents
all_tags = epie.tags_formal + epie.tags_static
#print(idiom_sent)

idiom_scorer = Scorer(model)

score_per_head = idiom_scorer.get_avg_idiom_scores(all_idiom_sents, all_tokenized_sents, all_tags)

if not os.path.isdir("./scores"):
    os.mkdir("./scores")

idiom_scorer.save_scores(score_per_head, f"./scores/{model_name.split('/')[-1]}_avg.json")

df = create_df(score_per_head)
plot_score_line(df)
plot_score_hist(df)