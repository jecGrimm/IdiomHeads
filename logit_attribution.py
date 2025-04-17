import torch as t
from transformer_lens import HookedTransformer, utils
from fancy_einsum import einsum
import numpy as np
import plotly.express as px

def imshow(tensor, xaxis="", yaxis="", caxis="", **kwargs):
    return px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)

def convert_tokens_to_string(tokens, batch_index=0):
    '''
    Helper function to convert tokens into a list of strings, for printing.
    '''
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{model.tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]

def plot_logit_attribution_sentence(logit_attr: t.Tensor, tokens: t.Tensor, title: str = "", xlabels = None):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(tokens)
    #x_labels = ["Direct"] + [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    # imshow(logit_attr.float(), x=x_labels, y=y_labels, xaxis="Term", yaxis="Position", caxis="logit", title=title if title else None, height=25*len(tokens))
    return imshow(logit_attr.float(), x=xlabels,  y=y_labels, xaxis="Component", yaxis="Position", caxis="logit", title=title if title else None, height=25*len(tokens))

def plot_logit_attribution_split(logit_attr: t.Tensor, title: str = "", x_labels = None):
    y_labels = ["Idiom", "Literal"]
    #return imshow(logit_attr.float(), x=x_labels, y=y_labels, xaxis="Component", yaxis="Position", caxis="mean logit", title=title if title else None, height=25*len(tokens))
    return px.imshow(logit_attr.float(), title = title, x=x_labels, y=y_labels, labels={"x":"Component", "y":"Position", "color":"Mean logit attribution"}, color_continuous_midpoint=0.0)

def logit_attribution(embed, l_results, W_U, tokens) -> t.Tensor:
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

    direct_attributions = einsum("emb seq, seq emb -> seq", W_U_correct_tokens, embed[:-1])
    concats = [direct_attributions.unsqueeze(-1)]

    for l_result in l_results:
        concats.append(einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l_result[:-1]))
    # l1_attributions = einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l1_results[:-1])
    # l2_attributions = einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l2_results[:-1])
    #return t.concat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)
    return t.concat(concats, dim=-1)

def split_logit_attribution(logit_attribution, idiom_pos):
    idiom_attr = mean_idiom_attribution(logit_attribution, idiom_pos)
    literal_attr = mean_literal_attribution(logit_attribution, idiom_pos)
    return t.vstack((idiom_attr, literal_attr))

def mean_idiom_attribution(logit_attr, idiom_pos):
    return t.mean(logit_attr[idiom_pos[0]:idiom_pos[1]], dim = 0)

def mean_literal_attribution(logit_attr, idiom_pos):
    return t.mean(t.cat((logit_attr[:idiom_pos[0]], logit_attr[idiom_pos[1]+1:])), dim = 0)

text = "‘Are not you going to spill the beans?"

#model = HookedTransformer.from_pretrained("EleutherAI/pythia-14m", dtype="bfloat16")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-14m")
model.cfg.use_attn_result = True
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    #embed = cache["embed"]

    # l1_results = cache["result", 0]
    # l2_results = cache["result", 1]
    #l_results = [cache["result", i] for i in range(model.cfg.n_layers)]
     
    #logit_attr = logit_attribution(embed, l_results, model.W_U, tokens[0])
    residual_stack, labels = cache.get_full_resid_decomposition(expand_neurons=False, return_labels=True)
    print(residual_stack)
    print(residual_stack.size())
    print(len(labels))
    logit_attr = cache.logit_attrs(residual_stack, tokens, has_batch_dim=False)
    token_first_attr = t.einsum("ij->ji", logit_attr)
    print("logit attr:", logit_attr)
    print(logit_attr.size())

    #logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    #correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    #t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)

# embed = cache["embed"]
# l1_results = cache["result", 0]
# l2_results = cache["result", 1]
# logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens[0])

# works jeij
#fig = plot_logit_attribution_sentence(token_first_attr, tokens, xlabels = labels)


#print(split_logit_attribution(logit_attr, (7, 9)))
split_logit_attr = split_logit_attribution(token_first_attr, (7, 9))
print("split_logit_attr:", split_logit_attr.size())
fig = plot_logit_attribution_split(split_logit_attr, title = "Mean logit attribution: Idiom vs. Literal", x_labels=labels)
#enable_plotly_in_cell()

#fig.write_image("./plots/pythia-14m/logit_attr.png")
fig.show()