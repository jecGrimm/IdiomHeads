from IPython.display import display
import circuitsvis as cv
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch as t
# %pip install einops
# %pip install jaxtyping
# %pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python

def plot_attention_heads(model, cache, str_tokens):
    for layer in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", layer]
        display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern, attention_head_names=[f"L{layer}H{i}" for i in range(16)]))

def create_df_from_dict(score_per_head: dict):
    return pd.DataFrame({"layer_heads": list(score_per_head.keys()), "scores": list(score_per_head.values())})

def create_df_from_tensor(tensor):
    return pd.DataFrame(tensor.numpy())

def plot_score_line(scores):
    df = create_df_from_dict(scores)
    df.plot.line(x="layer_heads", y="scores")
    plt.savefig("./plots/score_line.png")

def plot_score_hist(scores):
    df = create_df_from_dict(scores)
    df['scores'].plot(kind='hist', bins=20)
    #plt.gca().spines[['top', 'right',]].set_visible(False)
    plt.savefig("./plots/score_hist.png")   

def plot_dict_box(scores, filename):
    df = create_df_from_dict(scores)
    df.plot.box()

    save_plot(filename)

def plot_tensor_line(tensor, filename = None):
    df = create_df_from_tensor(tensor)
    df.plot.line()
    
    save_plot(filename)

def plot_tensor_hist(tensor, filename = None):
    flattened_tensor = tensor.view(tensor.size(0)*tensor.size(1))
    print(tensor.view(tensor.size(0)*tensor.size(1)).size())
    df = create_df_from_tensor(flattened_tensor)
    df.plot.hist()
    
    save_plot(filename)

def plot_tensor_box(tensor, filename = None):
    print(tensor.size())
    #flattened_tensor = tensor.view(tensor.size(0)*tensor.size(1))
    #print(flattened_tensor.size())
    #df = create_df_from_tensor(flattened_tensor)
    
    df = create_df_from_tensor(tensor)
    print(df)
    df.plot.box()
    
    save_plot(filename)

def get_mean_sentence_tensor(tensor):
    return t.mean(tensor, dim = 0)

def save_plot(filename: None):
    if filename:
        plt.savefig(filename) 
    else:
        plt.show()

# TODO: reshape tensor to get the values in form: layer.head = [sentence scores]
def get_lh_sentence_scores(tensor):
    scores = dict()
    sent_last = t.einsum("ijk->jki", tensor)
    for layer in range(tensor.size(1)):
        for head in range(tensor.size(2)):
            scores[f"{layer}.{head}"] = sent_last[layer][head]
    # 0.0: 0.9061, 0.9297, 0.9173
    # 0.1: 0.8422, 0.9493, 0.9504
    # 1.0: 0.9554, 0.8907, 0.8849
    return scores
    
if __name__ == "__main__":
    # with open("./scores_per_head.json", 'r', encoding="utf-8") as f:
    #     score_per_head = json.load(f)

    # df = create_df(score_per_head)
    # plot_score_hist(df)
    loaded_tensor = t.load("./scores/test_formal.pt")
    scores_dict = get_lh_sentence_scores(loaded_tensor)
    plot_dict_box(scores_dict)
    #plot_tensor_box(get_mean_tensor(loaded_tensor))
