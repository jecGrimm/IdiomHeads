from transformer_lens import (
    utils,
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

pythia_1_4b: HookedTransformer = HookedTransformer.from_pretrained("EleutherAI/pythia-1.4b")