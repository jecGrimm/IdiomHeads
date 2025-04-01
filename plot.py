from IPython.display import display
import circuitsvis as cv
import pandas as pd
import matplotlib.pyplot as plt
import json
# %pip install einops
# %pip install jaxtyping
# %pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python

def plot_attention_heads(model, cache, str_tokens):
    for layer in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", layer]
        display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern, attention_head_names=[f"L{layer}H{i}" for i in range(16)]))

def create_df(score_per_head: dict):
    return pd.DataFrame({"layer_heads": list(score_per_head.keys()), "scores": list(score_per_head.values())})

def plot_score_line(df):
    df.plot.line(x="layer_heads", y="scores")
    plt.savefig("./plots/score_line.png")

def plot_score_hist(df):
    df['scores'].plot(kind='hist', bins=20)
    #plt.gca().spines[['top', 'right',]].set_visible(False)
    plt.savefig("./plots/score_hist.png")
    
if __name__ == "__main__":
    with open("./scores_per_head.json", 'r', encoding="utf-8") as f:
        score_per_head = json.load(f)

    df = create_df(score_per_head)
    plot_score_hist(df)