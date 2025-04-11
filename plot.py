from IPython.display import display
import circuitsvis as cv
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch as t
import plotly.express as px

# %pip install jaxtyping
# %pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python

def plot_attention_heads(model, cache, str_tokens):
    for layer in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", layer]
        display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern, attention_head_names=[f"L{layer}H{i}" for i in range(16)]))

def create_df_from_dict(score_per_head: dict):
    return pd.DataFrame({"layer.head": list(score_per_head.keys()), "scores": list(score_per_head.values())}).set_index("layer.head")

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

def plot_box_per_head(tensor, filename = None):
    scores = get_lh_sentence_scores(tensor)
    df = create_df_from_dict(scores)
    df['scores'].apply(lambda x: pd.Series(x)).T.boxplot(figsize=(10,10),rot=45)

    save_plt(filename)

def plot_tensor_line(tensor, filename = None):
    mean_tensor = get_mean_sentence_tensor(tensor)
    df = create_df_from_tensor(mean_tensor)
    df.plot.line(title="Average idiom score per head and layer", xlabel = "Layer", ylabel = "Mean idiom score")
    plt.legend(title = "Head")
    save_plt(filename)

def plot_tensor_hist(tensor, filename = None):
    mean_tensor = get_mean_sentence_tensor(tensor)
    len = mean_tensor.size(0) * mean_tensor.size(1)
    df = create_df_from_tensor(mean_tensor.view(len))
    df.plot.hist(title="Distribution of the mean idiom scores", legend = False, xlabel="Mean idiom score")
    
    save_plt(filename)

def plot_box_avg(tensor, filename = None):
    print(tensor.size())
    #flattened_tensor = tensor.view(tensor.size(0)*tensor.size(1))
    #print(flattened_tensor.size())
    #df = create_df_from_tensor(flattened_tensor)
    
    df = create_df_from_tensor(tensor)
    print(df)
    df.plot.box()
    
    save_plt(filename)

def plot_heatmap(tensor, filename = None):
    mean_tensor = get_mean_sentence_tensor(tensor)
    #print(mean_tensor[0][0])
    fig = px.imshow(mean_tensor, labels=dict(x="Head", y="Layer"))

    if filename:
        fig.write_image(filename)
    else:
        fig.show()

def get_mean_sentence_tensor(tensor):
    return t.mean(tensor, dim = 0)

def save_plt(filename = None):
    if filename:
        plt.savefig(filename) 
    else:
        plt.show()

def get_lh_sentence_scores(tensor):
    scores = dict()
    sent_last = t.einsum("ijk->jki", tensor)
    for layer in range(tensor.size(1)):
        for head in range(tensor.size(2)):
            scores[f"{layer}.{head}"] = sent_last[layer][head].numpy()
    # 0.0: 0.9061, 0.9297, 0.9173
    # 0.1: 0.8422, 0.9493, 0.9504
    # 1.0: 0.9554, 0.8907, 0.8849
    return scores
    
def get_lh_mean_scores(tensor):
    scores = dict()
    sent_last = t.einsum("ijk->jki", tensor)
    for layer in range(tensor.size(1)):
        for head in range(tensor.size(2)):
            scores[f"{layer}.{head}"] = t.mean(sent_last[layer][head]).numpy()
    # 0.0: 0.9061, 0.9297, 0.9173
    # 0.1: 0.8422, 0.9493, 0.9504
    # 1.0: 0.9554, 0.8907, 0.8849
    return scores
    
if __name__ == "__main__":
    loaded_tensor = t.load("./scores/test_formal.pt")

    #plot_heatmap(loaded_tensor, "./plots/pythia_14m_mean_heat.png")
    #plot_line_mean_head(loaded_tensor)
    #plot_tensor_line(loaded_tensor, filename = "./plots/pythia_14m_mean_line.png")
    plot_tensor_hist(loaded_tensor, filename="./plots/pythia_14m_mean_hist.png")
    
