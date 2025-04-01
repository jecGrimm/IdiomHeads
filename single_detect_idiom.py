from transformer_lens import (
    HookedTransformer,
)
from data import EPIE_Data
from idiom_score import Scorer
from plot import plot_score_line

#pythia_1_4b: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-1.4b")
pythia_14m: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-14m")


# ATTENTION HEADS
# Single sentence
epie = EPIE_Data()
idx = 881
idiom_sent = epie.formal_sents[idx]
tokenized_idiom_sent = epie.tokenized_formal_sents[idx]
tags = epie.tags_formal[idx]
#print(idiom_sent)

idiom_tokens = pythia_14m.to_tokens(idiom_sent)
pythia_14m_logits, pythia_14m_cache = pythia_14m.run_with_cache(idiom_tokens, remove_batch_dim=True)
idiom_scorer = Scorer(pythia_14m, pythia_14m_cache, tokenized_idiom_sent, tags)

score_per_head = idiom_scorer.aggregate_components()
idiom_scorer.save_scores(score_per_head)
plot_score_line(score_per_head)