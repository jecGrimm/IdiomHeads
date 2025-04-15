from matplotlib.pyplot import imshow
import torch as t
from transformer_lens import HookedTransformer, utils

def logit_attribution(embed, l1_results, l2_results, W_U, tokens) -> t.Tensor:
    '''
    Inputs:
        embed (seq_len, d_model): the embeddings of the tokens (i.e. token + position embeddings)
        l1_results (seq_len, n_heads, d_model): the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results (seq_len, n_heads, d_model): the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U (d_model, d_vocab): the unembedding matrix
    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (position-1,1)
            layer 0 logits (position-1, n_heads)
            and layer 1 logits (position-1, n_heads)
    '''
    W_U_correct_tokens = W_U[:, tokens[1:]]

    direct_attributions = t.einsum("emb seq, seq emb -> seq", W_U_correct_tokens, embed[:-1])
    l1_attributions = t.einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l1_results[:-1])
    l2_attributions = t.einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l2_results[:-1])
    return t.concat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)

text = "'Arent you going to spill the beans?'"
# TODO: Fix set use_attn_result in the config to true
model = HookedTransformer.from_pretrained("EleutherAI/pythia-14m", dtype="bfloat16", use_attn_result=True)
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["attn_out", 0]
    l2_results = cache["attn_out", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)

def convert_tokens_to_string(tokens, batch_index=0):
    '''
    Helper function to convert tokens into a list of strings, for printing.
    '''
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{model.tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]

def plot_logit_attribution(logit_attr: t.Tensor, tokens: t.Tensor, title: str = ""):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(tokens[:-1])
    x_labels = ["Direct"] + [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    return imshow(utils.to_numpy(logit_attr), x=x_labels, y=y_labels, xaxis="Term", yaxis="Position", caxis="logit", title=title if title else None, height=25*len(tokens))

# embed = cache["embed"]
# l1_results = cache["result", 0]
# l2_results = cache["result", 1]
# logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens[0])
fig = plot_logit_attribution(logit_attr, tokens)
#enable_plotly_in_cell()
fig.show()