from IPython.display import display
import circuitsvis as cv
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch as t
import plotly.express as px
import argparse
import os
import numpy as np

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
    df['scores'].apply(lambda x: pd.Series(x)).T.boxplot(figsize=(100,100),rot=45)

    save_plt(filename)

def plot_tensor_line(tensor, filename = None):
    mean_tensor = get_mean_sentence_tensor(tensor)
    df = create_df_from_tensor(mean_tensor)
    df.plot.line(title="Average idiom score per head and layer", xlabel = "Layer", ylabel = "Mean idiom score", xticks = np.arange(mean_tensor.size(0)))
    plt.legend(title = "Head")
    save_plt(filename)

def plot_line_std(tensor, filename = None):
    std_tensor = t.std(tensor, dim=0)
    df = create_df_from_tensor(std_tensor)
    #print("std:", std_tensor.size())
    df.plot.line(title="Standard deviation of the idiom score per head and layer", xlabel = "Layer", ylabel = "Standard deviation of the idiom score", xticks = np.arange(std_tensor.size(0)))
    plt.legend(title = "Head")
    save_plt(filename)

def plot_tensor_hist(tensor, filename = None):
    mean_tensor = get_mean_sentence_tensor(tensor)
    len = mean_tensor.size(0) * mean_tensor.size(1)
    df = create_df_from_tensor(mean_tensor.view(len))
    df.plot.hist(title="Distribution of the mean idiom scores", legend = False, xlabel="Mean idiom score")
    
    save_plt(filename)

def plot_box_avg(tensor, filename = None):
    #print(tensor.size())
    #flattened_tensor = tensor.view(tensor.size(0)*tensor.size(1))
    #print(flattened_tensor.size())
    #df = create_df_from_tensor(flattened_tensor)
    
    df = create_df_from_tensor(tensor)
    #print(df)
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

def plot_all(tensor, filename = None, model_name = None):
    path = f"./plots/{model_name}"
    if filename and model_name:
        if not os.path.isdir("./plots"):
            os.mkdir("./plots")

        if not os.path.isdir(f"./plots/{model_name}"):
            os.mkdir(f"./plots/{model_name}")

        plot_tensor_line(tensor, f"{path}/mean_line_{filename}.png")
        plot_line_std(tensor, f"{path}/std_line_{filename}.png")
        plot_heatmap(tensor, f"{path}/heat_{filename}.png")
        plot_tensor_hist(tensor, f"{path}/hist_{filename}.png")
    else:
        plot_tensor_line(tensor)
        plot_line_std(tensor)
        plot_heatmap(tensor)
        plot_tensor_hist(tensor)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='idiom head detector')
    parser.add_argument('--model_name', '-m', help='model to run the experiment with', default="pythia-1.4b")
    parser.add_argument('--tensorfile', '-t', help='file with the tensor scores', default="./components/idiom_components/pythia-1.4b/idiom_only_formal_0_None.pt", type=str)
    parser.add_argument('--imagefile', '-i', help='output file for the plot', default=None, type=str)

    model_name = parser.parse_args().model_name
    tensor_file = parser.parse_args().tensorfile
    img_file = parser.parse_args().imagefile
    device = "cuda" if t.cuda.is_available() else "cpu"

    loaded_tensor = t.load(tensor_file, map_location=t.device(device))
    print(f"Loaded tensor with size: {loaded_tensor.size()}")

    plot_all(loaded_tensor, img_file, model_name)

    #plot_tensor_line(loaded_tensor, f"./plots/{model_name}/line_idiom_only_formal.png")
    # plot_box_avg(loaded_tensor)
    #plot_box_per_head(loaded_tensor)
    #plot_line_std(loaded_tensor)


    #plot_heatmap(loaded_tensor, "./plots/pythia_14m_mean_heat.png")
    
    #plot_tensor_hist(loaded_tensor, filename="./plots/pythia_14m_mean_hist.png")
    
