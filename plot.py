import pandas as pd
import matplotlib.pyplot as plt
import torch as t
import plotly.express as px
import argparse
import os
import numpy as np
from collections import defaultdict
from helper import get_logit_component
import json
from data import EPIE_Data
import seaborn as sns

def create_df_from_dict(score_per_head: dict):
    """
    This function transforms a dictionary into a pandas Dataframe.

    @params
        score_per_head: score dictionary
    @returns
        dataframe containing the scores 
    """
    return pd.DataFrame({"layer.head": list(score_per_head.keys()), "scores": list(score_per_head.values())}).set_index("layer.head")

def create_df_from_tensor(tensor):
    """
    This function transforms a tensor into a pandas Dataframe.

    @params
        tensor: tensor to transform
    @returns
        dataframe containing the tensor values 
    """
    return pd.DataFrame(tensor.numpy())

def plot_loss(tensor, filename: str = None, model_name: str = None):
    """
    This function visualizes the loss scores.

    @params
        tensor: tensor with loss scores
        filename: name of the output file without file ending
        model_name: model ID
    """
    line_file, hist_file, box_file, txt_file = None, None, None, None
    if filename != None and model_name != None:
        path = f"./plots/{model_name}/loss/"
        os.makedirs(path, exist_ok=True)
        line_file = path + "line_" +filename + ".png"
        hist_file = path + filename + "_hist.png"
        box_file = path + filename + "_box.png"
        txt_file = path+filename+".txt"
 
    df = create_df_from_tensor(tensor)
    df.plot.line(legend=False, title= "Loss per Sentence", xlabel= "Sentence ID", ylabel = "Loss")
    save_plt(line_file)

    df.plot.hist(bins=20, title= "Loss Distribution", xlabel= "Loss", ylabel = "Frequency", legend=False)
    save_plt(hist_file)

    df.plot.box(title= "Loss Boxplot", xticks= [], ylabel = "Loss")
    save_plt(box_file)

    epie = EPIE_Data()

    output = f"Mean loss: {tensor.mean()}\n"

    if txt_file:
        if "formal" in filename:
            output += f"Sentence with the minimum loss {tensor.min()}:\n{epie.formal_sents[tensor.argmin().int()]}"
            output += f"\nSentence with the maximum loss {tensor.max()}:\n{epie.formal_sents[tensor.argmax().int()]}"
        elif "trans" in filename:
            output += f"Sentence with the minimum loss {tensor.min()}:\n{epie.trans_formal_sents[tensor.argmin().int()]}"
            output += f"\nSentence with the maximum loss {tensor.max()}:\n{epie.trans_formal_sents[tensor.argmax().int()]}"
        else: 
            output += f"Sentence with the minimum loss {tensor.min()}"
            output += f"\nSentence with the maximum loss {tensor.max()}"

        with open(txt_file, 'w', encoding="utf-8") as f:
            f.write(output)
    else: 
        output += f"Minimum loss {tensor.min()}"
        output += f"\nMaximum loss {tensor.max()}"
        print(output) 

def plot_tensor_dots(tensor, filename: str = None, title: str = "Average score per head and layer", ylabel: str = "Mean score"):
    """
    This function plots averaged scores per head.

    @params
        tensor: scores per head
        filename: name of the output file with file ending
        title: plot title
        ylabel: label of the y-axis
    """
    mean_tensor = get_mean_sentence_tensor(tensor)
    df = create_df_from_tensor(mean_tensor)

    ax = df.plot(marker= '.', ms = 10, linestyle='none', title= title, xlabel = "Layer", ylabel = ylabel, xticks = np.arange(mean_tensor.size(0)), colormap = "tab20", legend = True)
    
    # adjust legend to fit all heads in the frame 
    ax.legend(ncol=len(df.columns)/4, bbox_to_anchor=(0.5, -0.45), title = "Head", loc='lower center', fontsize=6)
    plt.subplots_adjust(bottom=0.3, top=0.91)

    save_plt(filename, dpi = 300, bbox_inches='tight')

def plot_std_dot(tensor, filename: str = None, title: str = "Standard deviation of the score per head and layer", ylabel: str = "Standard deviation of the score"):
    """
    This function plots the std of the scores per head.

    @params
        tensor: scores per head
        filename: name of the output file with file ending
        title: plot title
        ylabel: label of the y-axis
    """
    std_tensor = t.std(tensor, dim=0)
    df = create_df_from_tensor(std_tensor)

    ax = df.plot(marker= '.', ms=10, linestyle='none', title= title, xlabel = "Layer", ylabel = ylabel, xticks = np.arange(std_tensor.size(0)), colormap = "tab20", legend = True)
    ax.legend(ncol=len(df.columns)/4, bbox_to_anchor=(0.5, -0.45), title = "Head", loc='lower center', fontsize=6)
    plt.subplots_adjust(bottom=0.3, top=0.91)

    save_plt(filename)

def plot_tensor_hist(tensor, filename: str = None, title: str = "Distribution of the mean scores", xlabel: str = "Mean scores"):
    """
    This function creates a histogram of the scores per head.

    @params
        tensor: scores per head
        filename: name of the output file with file ending
        title: plot title
        xlabel: label of the x-axis
    """
    mean_tensor = get_mean_sentence_tensor(tensor)
    len = mean_tensor.size(0) * mean_tensor.size(1)
    df = create_df_from_tensor(mean_tensor.view(len))
    df.plot.hist(title=title, legend = False, xlabel=xlabel) 
    
    save_plt(filename)

def plot_heatmap(tensor, filename: str = None, title: str = "Scores"):
    """
    This function creates a heatmap of the averaged scores per head.

    @params
        tensor: scores per head and sentence
        filename: name of the output file with file ending
        title: plot title
    """
    mean_tensor = get_mean_sentence_tensor(tensor)
    fig = px.imshow(mean_tensor, labels=dict(x="Head", y="Layer"), title = title)

    if filename:
        fig.write_image(filename)
    else:
        fig.show()

def plot_scatter(idiom_tensor, literal_tensor, filename: str = None, xlabel: str = "Idiom Score", ylabel: str = "Literal Score"):
    """
    This function creates a scatter plot for the Idiom and Literal Scores.

    @params
        idiom_tensor: x-axis values
        literal_tensor: y-axis values
        filename: name of the output file with file ending
        xlabel: label of the x-axis
        ylabel: label of the y-axis
    """
    idiom_mean = get_lh_mean_scores(idiom_tensor) 
    literal_mean = get_lh_mean_scores(literal_tensor) 

    df = pd.DataFrame({"layer.head": list(idiom_mean.keys()), "idiom_mean": list(idiom_mean.values()), "literal_mean": list(literal_mean.values())})
    df.plot.scatter(x='idiom_mean',y='literal_mean', xlabel= xlabel, ylabel = ylabel)

    save_plt(filename)

def plot_scatter_idiom_logit(model_name: str, filename: str = None):
    """
    This function creates a scatter plot for the Idiom and DLA Scores.

    @params
        model_name: model ID
        filename: name of the output file with file ending
    """
    device = "cuda" if t.cuda.is_available() else "cpu"

    x_labels = {
        "pythia-14m": [["L0H0", "L0H1", "L0H2", "L0H3", "L1H0", "L1H1", "L1H2", "L1H3", "L2H0", "L2H1", "L2H2", "L2H3", "L3H0", "L3H1", "L3H2", "L3H3", "L4H0", "L4H1", "L4H2", "L4H3", "L5H0", "L5H1", "L5H2", "L5H3", "0_mlp_out", "1_mlp_out", "2_mlp_out", "3_mlp_out", "4_mlp_out", "5_mlp_out", "embed", "bias"]],
        "Pythia-1.4B": [['L0H0', 'L0H1', 'L0H2', 'L0H3', 'L0H4', 'L0H5', 'L0H6', 'L0H7', 'L0H8', 'L0H9', 'L0H10', 'L0H11', 'L0H12', 'L0H13', 'L0H14', 'L0H15'], ['L1H0', 'L1H1', 'L1H2', 'L1H3', 'L1H4', 'L1H5', 'L1H6', 'L1H7', 'L1H8', 'L1H9', 'L1H10', 'L1H11', 'L1H12', 'L1H13', 'L1H14', 'L1H15'], ['L2H0', 'L2H1', 'L2H2', 'L2H3', 'L2H4', 'L2H5', 'L2H6', 'L2H7', 'L2H8', 'L2H9', 'L2H10', 'L2H11', 'L2H12', 'L2H13', 'L2H14', 'L2H15'], ['L3H0', 'L3H1', 'L3H2', 'L3H3', 'L3H4', 'L3H5', 'L3H6', 'L3H7', 'L3H8', 'L3H9', 'L3H10', 'L3H11', 'L3H12', 'L3H13', 'L3H14', 'L3H15'], ['L4H0', 'L4H1', 'L4H2', 'L4H3', 'L4H4', 'L4H5', 'L4H6', 'L4H7', 'L4H8', 'L4H9', 'L4H10', 'L4H11', 'L4H12', 'L4H13', 'L4H14', 'L4H15'], ['L5H0', 'L5H1', 'L5H2', 'L5H3', 'L5H4', 'L5H5', 'L5H6', 'L5H7', 'L5H8', 'L5H9', 'L5H10', 'L5H11', 'L5H12', 'L5H13', 'L5H14', 'L5H15'], ['L6H0', 'L6H1', 'L6H2', 'L6H3', 'L6H4', 'L6H5', 'L6H6', 'L6H7', 'L6H8', 'L6H9', 'L6H10', 'L6H11', 'L6H12', 'L6H13', 'L6H14', 'L6H15'], ['L7H0', 'L7H1', 'L7H2', 'L7H3', 'L7H4', 'L7H5', 'L7H6', 'L7H7', 'L7H8', 'L7H9', 'L7H10', 'L7H11', 'L7H12', 'L7H13', 'L7H14', 'L7H15'], ['L8H0', 'L8H1', 'L8H2', 'L8H3', 'L8H4', 'L8H5', 'L8H6', 'L8H7', 'L8H8', 'L8H9', 'L8H10', 'L8H11', 'L8H12', 'L8H13', 'L8H14', 'L8H15'], ['L9H0', 'L9H1', 'L9H2', 'L9H3', 'L9H4', 'L9H5', 'L9H6', 'L9H7', 'L9H8', 'L9H9', 'L9H10', 'L9H11', 'L9H12', 'L9H13', 'L9H14', 'L9H15'], ['L10H0', 'L10H1', 'L10H2', 'L10H3', 'L10H4', 'L10H5', 'L10H6', 'L10H7', 'L10H8', 'L10H9', 'L10H10', 'L10H11', 'L10H12', 'L10H13', 'L10H14', 'L10H15'], ['L11H0', 'L11H1', 'L11H2', 'L11H3', 'L11H4', 'L11H5', 'L11H6', 'L11H7', 'L11H8', 'L11H9', 'L11H10', 'L11H11', 'L11H12', 'L11H13', 'L11H14', 'L11H15'], ['L12H0', 'L12H1', 'L12H2', 'L12H3', 'L12H4', 'L12H5', 'L12H6', 'L12H7', 'L12H8', 'L12H9', 'L12H10', 'L12H11', 'L12H12', 'L12H13', 'L12H14', 'L12H15'], ['L13H0', 'L13H1', 'L13H2', 'L13H3', 'L13H4', 'L13H5', 'L13H6', 'L13H7', 'L13H8', 'L13H9', 'L13H10', 'L13H11', 'L13H12', 'L13H13', 'L13H14', 'L13H15'], ['L14H0', 'L14H1', 'L14H2', 'L14H3', 'L14H4', 'L14H5', 'L14H6', 'L14H7', 'L14H8', 'L14H9', 'L14H10', 'L14H11', 'L14H12', 'L14H13', 'L14H14', 'L14H15'], ['L15H0', 'L15H1', 'L15H2', 'L15H3', 'L15H4', 'L15H5', 'L15H6', 'L15H7', 'L15H8', 'L15H9', 'L15H10', 'L15H11', 'L15H12', 'L15H13', 'L15H14', 'L15H15'], ['L16H0', 'L16H1', 'L16H2', 'L16H3', 'L16H4', 'L16H5', 'L16H6', 'L16H7', 'L16H8', 'L16H9', 'L16H10', 'L16H11', 'L16H12', 'L16H13', 'L16H14', 'L16H15'], ['L17H0', 'L17H1', 'L17H2', 'L17H3', 'L17H4', 'L17H5', 'L17H6', 'L17H7', 'L17H8', 'L17H9', 'L17H10', 'L17H11', 'L17H12', 'L17H13', 'L17H14', 'L17H15'], ['L18H0', 'L18H1', 'L18H2', 'L18H3', 'L18H4', 'L18H5', 'L18H6', 'L18H7', 'L18H8', 'L18H9', 'L18H10', 'L18H11', 'L18H12', 'L18H13', 'L18H14', 'L18H15'], ['L19H0', 'L19H1', 'L19H2', 'L19H3', 'L19H4', 'L19H5', 'L19H6', 'L19H7', 'L19H8', 'L19H9', 'L19H10', 'L19H11', 'L19H12', 'L19H13', 'L19H14', 'L19H15'], ['L20H0', 'L20H1', 'L20H2', 'L20H3', 'L20H4', 'L20H5', 'L20H6', 'L20H7', 'L20H8', 'L20H9', 'L20H10', 'L20H11', 'L20H12', 'L20H13', 'L20H14', 'L20H15'], ['L21H0', 'L21H1', 'L21H2', 'L21H3', 'L21H4', 'L21H5', 'L21H6', 'L21H7', 'L21H8', 'L21H9', 'L21H10', 'L21H11', 'L21H12', 'L21H13', 'L21H14', 'L21H15'], ['L22H0', 'L22H1', 'L22H2', 'L22H3', 'L22H4', 'L22H5', 'L22H6', 'L22H7', 'L22H8', 'L22H9', 'L22H10', 'L22H11', 'L22H12', 'L22H13', 'L22H14', 'L22H15'], ['L23H0', 'L23H1', 'L23H2', 'L23H3', 'L23H4', 'L23H5', 'L23H6', 'L23H7', 'L23H8', 'L23H9', 'L23H10', 'L23H11', 'L23H12', 'L23H13', 'L23H14', 'L23H15'], ['0_mlp_out', '1_mlp_out', '2_mlp_out', '3_mlp_out', '4_mlp_out', '5_mlp_out', '6_mlp_out', '7_mlp_out', '8_mlp_out', '9_mlp_out', '10_mlp_out', '11_mlp_out', '12_mlp_out', '13_mlp_out', '14_mlp_out', '15_mlp_out', '16_mlp_out', '17_mlp_out', '18_mlp_out', '19_mlp_out', '20_mlp_out', '21_mlp_out', '22_mlp_out', '23_mlp_out'], ['embed', 'bias']],
        "Llama-3.2-1B-Instruct" : [[f"L{i}H{j}" for j in range(32)] for i in range(16)] + [['0_mlp_out', '1_mlp_out', '2_mlp_out', '3_mlp_out', '4_mlp_out', '5_mlp_out', '6_mlp_out', '7_mlp_out', '8_mlp_out', '9_mlp_out', '10_mlp_out', '11_mlp_out', '12_mlp_out', '13_mlp_out', '14_mlp_out', '15_mlp_out'], ['embed', 'bias']]
    }

    components = []
    for comp_group in x_labels[model_name]:
        for comp in comp_group:
            components.append(comp)

    # DLA
    # Formal
    logit_formal_file = f"scores/logit_attribution/{model_name}/grouped_attr_formal_0_None.pt"
    formal_logits = t.load(logit_formal_file, map_location=t.device(device))
    mean_formal_logits = get_mean_sentence_tensor(formal_logits)
    df_formal_attr = pd.DataFrame(mean_formal_logits.numpy().T, index = components, columns=['DLA Idiom Formal', 'DLA Literal Formal'])
    df_formal_attr_heads = df_formal_attr.filter(regex=r"L\d+H\d+", axis="index")

    idiom_tensor = t.load(f"scores/idiom_scores/{model_name}/idiom_formal_0_None.pt", map_location=t.device(device))
    idiom_mean = get_lh_mean_scores(idiom_tensor) # dict layer.head = mean over all sentences

    df_idiom = pd.DataFrame({"layer.head": list(idiom_mean.keys()), "idiom_mean": list(idiom_mean.values())}).set_index("layer.head")
    
    df = df_formal_attr_heads.join(df_idiom)
    df.plot.scatter(x='idiom_mean',y='DLA Idiom Formal', xlabel= "Idiom Score", ylabel = "Direct Logit Attribution", title = "Scatter plot for DLA and Idiom Scores")

    save_plt(filename)

def plot_scatter_components(comp_dict: dict, filename: str = None):
    """
    This function creates a scatter plot for the components of the Idiom Score.

    @params
        comp_dict: dictionary with the component scores
        filename: name of the output file with file ending
    """
    df = pd.DataFrame()
    for comp, tensor in comp_dict.items():
        comp_lh = get_lh_mean_scores(tensor)
        comp_df = pd.DataFrame({"layer.head": list(comp_lh.keys()), comp: list(comp_lh.values())}).set_index("layer.head")
        if df.empty:
            df = comp_df
        else:
            df = df.join(comp_df, on='layer.head')

    fig, axes = plt.subplots(nrows = 5, ncols = 5, constrained_layout=True, figsize=(20,20))
    comps = list(comp_dict.keys())
    for i in range(5):
        for j in range(5):
            axes[i,j].set_title(f"{comps[i]}/{comps[j]}")
            axes[i,j].scatter(df[comps[i]],df[comps[j]])

    save_plt(filename)

def get_mean_sentence_tensor(tensor):
    """
    This function computes the mean of the tensor values.

    @params
        tensor: tensor that is averaged
    @returns
        mean tensor
    """
    return t.mean(tensor, dim = 0)

def save_plt(filename: str = None, dpi: int = None, bbox_inches: str = None):
    """
    This function saves a plot.

    @params
        filename: output file with file ending
        dpi: dpi of the saved plot
        bbox_inches: bbox_inches of the saved plot
    """
    if filename:
        if dpi: 
            if bbox_inches:
                plt.savefig(filename, dpi = dpi, bbox_inches = bbox_inches)
            else:
                plt.savefig(filename, dpi = dpi)            
        else:
            if bbox_inches:
                plt.savefig(filename, bbox_inches = bbox_inches)
            else:
                plt.savefig(filename)    
    else:
        plt.show()
    
def get_lh_mean_scores(tensor):
    """
    This function transforms a tensor into a dictionary containing the scores per head.

    @params
        tensor: transformed tensor
    @returns
        scores: dictionary with the scores per head
    """
    scores = dict()
    sent_last = t.einsum("ijk->jki", tensor)
    for layer in range(tensor.size(1)):
        for head in range(tensor.size(2)):
            scores[f"L{layer}H{head}"] = t.mean(sent_last[layer][head]).numpy()
    return scores

def get_lh_std_scores(tensor):
    """
    This function transforms a tensor into a dictionary containing the std per head.

    @params
        tensor: transformed tensor
    @returns
        scores: dictionary with the std per head
    """
    scores = dict()
    sent_last = t.einsum("ijk->jki", tensor)
    for layer in range(tensor.size(1)):
        for head in range(tensor.size(2)):
            scores[f"L{layer}H{head}"] = t.std(sent_last[layer][head]).numpy()
    return scores

def explore_scores(tensor, filename: str = None):
    """
    This function collects information about the tensor values.

    @params
        tensor: tensor with inspected values
        filename: output file with file ending txt
    """
    output = ""
    # MEAN
    mean_scores = get_lh_mean_scores(tensor)
    output += f"\nAverage score: {t.mean(tensor)}"
    output += f"\nMaximum mean: {max(mean_scores, key=lambda k:mean_scores.get(k))} - {max(mean_scores.values())}"
    output += f"\nMinimum mean: {min(mean_scores, key=lambda k:mean_scores.get(k))} - {min(mean_scores.values())}"

    mean_top_10 = [(head, mean_scores.get(head)) for head in sorted(mean_scores, key = lambda k:mean_scores.get(k), reverse = True)][:10] 
    mean_bottom_10 = [(head, mean_scores.get(head)) for head in sorted(mean_scores, key = lambda k:mean_scores.get(k))][:10]
    output += f"\nMean top 10: {mean_top_10}"
    output += f"\nMean bottom 10: {mean_bottom_10}"

    # STD
    std_scores = get_lh_std_scores(tensor)
    output += f"\n\nMaximum std: {max(std_scores, key=lambda k:std_scores.get(k))} - {max(std_scores.values())}"
    output += f"\nMinimum std: {min(std_scores, key=lambda k:std_scores.get(k))} - {min(std_scores.values())}"

    std_top_10 = [(head, std_scores.get(head)) for head in sorted(std_scores, key = lambda k:std_scores.get(k), reverse = True)][:10] 
    std_bottom_10 = [(head, std_scores.get(head)) for head in sorted(std_scores, key = lambda k:std_scores.get(k))][:10]
    output += f"\nStd top 10: {std_top_10}"
    output += f"\nStd bottom 10: {std_bottom_10}"

    if filename != None:
        with open(filename, 'w', encoding = "utf-8") as f:
            f.write(output)
    else:
        print(output)

def get_head_info(layer_head, tensor):
    """
    This function prints the score for a single head.

    @params
        layer_head: head-ID in the form L0H0
        tensor: tensor containing the value of the head        
    """
    mean_scores = get_lh_mean_scores(tensor)
    ranked_mean = list(create_df_from_dict(mean_scores).sort_values(by="scores", ascending = False).index)
    std_scores = get_lh_std_scores(tensor)

    print(f"\nHead: {layer_head}\n\tScore: {mean_scores[layer_head]}\n\tStd: {std_scores[layer_head]}\n\tRank: {ranked_mean.index(layer_head)+1} of {len(ranked_mean)}")

def get_logit_info(num: int, tensor, model_name: str):
    """
    This function pronts the DLA score for a single head.

    @params
        num: number of the head
        tensor: DLA scores
        model_name: model ID 
    """
    mean_logit_attr = get_mean_sentence_tensor(tensor)
    print(f"\nNumber {num}, Component {get_logit_component(num, model_name)}\nIdiom Logit: {mean_logit_attr[num][0]}\nLiteral Logit: {mean_logit_attr[num][1]}")

def plot_logit_attribution_split(logit_attr, title: str = "", x_labels: list = None, filename: str = None):
    """
    This function plots the DLA scores.

    @params
        logit_attr: tensor with DLA scores
        title: title of the plot
        x_labels: labels of the x-axis
        y_labels: labels of the y-axis
        filename: filename with png-ending
    """
    y_labels = ["Idiom", "Literal"]

    fig = px.imshow(logit_attr.float(), title = title, x=x_labels, y=y_labels, labels={"x":"Component", "y":"Position", "color":"Mean Logit Attribution"}, color_continuous_midpoint=0.0)

    if filename:
        fig.write_image(filename)
    else:
        fig.show()

def plot_idiom_scores(tensor, filename: str = None, model_name: str = None, scatter_file: str = None):
    """
    This function plots the Idiom Scores.

    @params
        tensor: tensor with Idiom Scores
        filename: filename with png-ending
        model_name: model ID
        scatter_file: file with Literal or DLA scores
    """
    if filename:
        if "idiom" in filename: 
            path = f"./plots/{model_name}/idiom_scores"

            if "static" in filename:
                path = f"./future_work/plots/{model_name}/idiom_scores"
        elif "literal" in filename:
            path = f"./plots/{model_name}/literal_scores"

            if "static" in filename:
                path = f"./future_work/plots/{model_name}/literal_scores"
        elif "attr" in filename:
            path = f"./plots/{model_name}/logit_attribution"
        else:
            path = f"./plots/{model_name}/scores"
        os.makedirs(path, exist_ok=True)

    if filename and model_name:
        plot_tensor_dots(tensor, f"{path}/mean_{filename}.png")
        plot_std_dot(tensor, f"{path}/std_{filename}.png")
        plot_heatmap(tensor, f"{path}/heat_{filename}.png" )
        plot_tensor_hist(tensor, f"{path}/hist_{filename}.png")
        explore_scores(tensor, f"{path}/{filename}.txt")
    else:
        plot_tensor_dots(tensor)
        plot_std_dot(tensor)
        plot_heatmap(tensor)
        plot_tensor_hist(tensor)
        explore_scores(tensor)

    if scatter_file != None:
        if "literal" in scatter_file:
            device = "cuda" if t.cuda.is_available() else "cpu"
            scatter_tensor = t.load(scatter_file, map_location=t.device(device))
            print(f"Loaded tensor with size: {scatter_tensor.size()}")

            if filename and model_name:
                plot_scatter(tensor, scatter_tensor, f"{path}/scatter_{filename}_literal_{model_name}.png")
            else:
                plot_scatter(tensor, scatter_tensor)
        elif "attr" in scatter_file and model_name:
            plot_scatter_idiom_logit(model_name=model_name, filename=f"{path}/scatter_{filename}_logit_{model_name}.png")
    
def get_component_dict(tensor):
    """
    This function creates the component dictionary.

    @params
        tensor: component tensor
    @returns 
        comp_dict: component dict
    """
    comp_dict = {
        "Mean": tensor[:, :, :, 0],
        "Std": tensor[:, :, :, 1],
        "Max": tensor[:, :, :, 2],
        "Phrase": tensor[:, :, :, 3],
        "Contribution": tensor[:, :, :, 4] 
    }
    return comp_dict

def plot_all_components(full_tensor, filename: str = None, model_name: str = None, path: str = ""):
    """
    This function plots the components of the Idiom Scores.

    @params
        full_tensor: tensor with the component scores per sentence and head
        filename: filename without ending
        model_name: model ID
    """
    comp_dict = get_component_dict(full_tensor)
    orig_filename = filename
    for comp, tensor in comp_dict.items():
        if orig_filename and model_name:
            filename = orig_filename + f"_{comp}"
            plot_tensor_dots(tensor, f"{path}/mean_line_{filename}.png", title = f"Average {comp} (component) per head and layer", ylabel=comp)
            plot_std_dot(tensor, f"{path}/std_line_{filename}.png", title = f"Standard deviation of the {comp} (component) per head and layer", ylabel = comp)
            plot_heatmap(tensor, f"{path}/heat_{filename}.png", title = f"{comp[0].upper()}{comp[1:]} (component)")
            plot_tensor_hist(tensor, f"{path}/hist_{filename}.png", title = f"Distribution of the {comp} (component)", xlabel = comp)
            explore_scores(tensor, f"{path}/{filename}.txt")

        else:
            plot_tensor_dots(tensor, title = f"Average {comp} (component) per head and layer", ylabel=comp)
            plot_std_dot(tensor, title = f"Standard deviation of the {comp} (component) per head and layer", ylabel=comp)
            plot_heatmap(tensor, title = f"{comp[0].upper()}{comp[1:]} (component)")
            plot_tensor_hist(tensor, title = f"Distribution of the {comp} (component)", xlabel=comp)
            print(f"\n\nComponent: {comp}")
            explore_scores(tensor)

    if orig_filename and model_name:
        plot_scatter_components(comp_dict, f"{path}/scatter_{orig_filename}_comp.png")
    else:
        plot_scatter_components(comp_dict)

def create_csv(model_name: str, device):
    """
    This function stores all scores in a csv-file.

    @params
        model_name: model ID
        device: device (gpu or cpu)
    """
    x_labels = {
        "pythia-14m": [["L0H0", "L0H1", "L0H2", "L0H3", "L1H0", "L1H1", "L1H2", "L1H3", "L2H0", "L2H1", "L2H2", "L2H3", "L3H0", "L3H1", "L3H2", "L3H3", "L4H0", "L4H1", "L4H2", "L4H3", "L5H0", "L5H1", "L5H2", "L5H3", "0_mlp_out", "1_mlp_out", "2_mlp_out", "3_mlp_out", "4_mlp_out", "5_mlp_out", "embed", "bias"]],
        "Pythia-1.4B": [['L0H0', 'L0H1', 'L0H2', 'L0H3', 'L0H4', 'L0H5', 'L0H6', 'L0H7', 'L0H8', 'L0H9', 'L0H10', 'L0H11', 'L0H12', 'L0H13', 'L0H14', 'L0H15'], ['L1H0', 'L1H1', 'L1H2', 'L1H3', 'L1H4', 'L1H5', 'L1H6', 'L1H7', 'L1H8', 'L1H9', 'L1H10', 'L1H11', 'L1H12', 'L1H13', 'L1H14', 'L1H15'], ['L2H0', 'L2H1', 'L2H2', 'L2H3', 'L2H4', 'L2H5', 'L2H6', 'L2H7', 'L2H8', 'L2H9', 'L2H10', 'L2H11', 'L2H12', 'L2H13', 'L2H14', 'L2H15'], ['L3H0', 'L3H1', 'L3H2', 'L3H3', 'L3H4', 'L3H5', 'L3H6', 'L3H7', 'L3H8', 'L3H9', 'L3H10', 'L3H11', 'L3H12', 'L3H13', 'L3H14', 'L3H15'], ['L4H0', 'L4H1', 'L4H2', 'L4H3', 'L4H4', 'L4H5', 'L4H6', 'L4H7', 'L4H8', 'L4H9', 'L4H10', 'L4H11', 'L4H12', 'L4H13', 'L4H14', 'L4H15'], ['L5H0', 'L5H1', 'L5H2', 'L5H3', 'L5H4', 'L5H5', 'L5H6', 'L5H7', 'L5H8', 'L5H9', 'L5H10', 'L5H11', 'L5H12', 'L5H13', 'L5H14', 'L5H15'], ['L6H0', 'L6H1', 'L6H2', 'L6H3', 'L6H4', 'L6H5', 'L6H6', 'L6H7', 'L6H8', 'L6H9', 'L6H10', 'L6H11', 'L6H12', 'L6H13', 'L6H14', 'L6H15'], ['L7H0', 'L7H1', 'L7H2', 'L7H3', 'L7H4', 'L7H5', 'L7H6', 'L7H7', 'L7H8', 'L7H9', 'L7H10', 'L7H11', 'L7H12', 'L7H13', 'L7H14', 'L7H15'], ['L8H0', 'L8H1', 'L8H2', 'L8H3', 'L8H4', 'L8H5', 'L8H6', 'L8H7', 'L8H8', 'L8H9', 'L8H10', 'L8H11', 'L8H12', 'L8H13', 'L8H14', 'L8H15'], ['L9H0', 'L9H1', 'L9H2', 'L9H3', 'L9H4', 'L9H5', 'L9H6', 'L9H7', 'L9H8', 'L9H9', 'L9H10', 'L9H11', 'L9H12', 'L9H13', 'L9H14', 'L9H15'], ['L10H0', 'L10H1', 'L10H2', 'L10H3', 'L10H4', 'L10H5', 'L10H6', 'L10H7', 'L10H8', 'L10H9', 'L10H10', 'L10H11', 'L10H12', 'L10H13', 'L10H14', 'L10H15'], ['L11H0', 'L11H1', 'L11H2', 'L11H3', 'L11H4', 'L11H5', 'L11H6', 'L11H7', 'L11H8', 'L11H9', 'L11H10', 'L11H11', 'L11H12', 'L11H13', 'L11H14', 'L11H15'], ['L12H0', 'L12H1', 'L12H2', 'L12H3', 'L12H4', 'L12H5', 'L12H6', 'L12H7', 'L12H8', 'L12H9', 'L12H10', 'L12H11', 'L12H12', 'L12H13', 'L12H14', 'L12H15'], ['L13H0', 'L13H1', 'L13H2', 'L13H3', 'L13H4', 'L13H5', 'L13H6', 'L13H7', 'L13H8', 'L13H9', 'L13H10', 'L13H11', 'L13H12', 'L13H13', 'L13H14', 'L13H15'], ['L14H0', 'L14H1', 'L14H2', 'L14H3', 'L14H4', 'L14H5', 'L14H6', 'L14H7', 'L14H8', 'L14H9', 'L14H10', 'L14H11', 'L14H12', 'L14H13', 'L14H14', 'L14H15'], ['L15H0', 'L15H1', 'L15H2', 'L15H3', 'L15H4', 'L15H5', 'L15H6', 'L15H7', 'L15H8', 'L15H9', 'L15H10', 'L15H11', 'L15H12', 'L15H13', 'L15H14', 'L15H15'], ['L16H0', 'L16H1', 'L16H2', 'L16H3', 'L16H4', 'L16H5', 'L16H6', 'L16H7', 'L16H8', 'L16H9', 'L16H10', 'L16H11', 'L16H12', 'L16H13', 'L16H14', 'L16H15'], ['L17H0', 'L17H1', 'L17H2', 'L17H3', 'L17H4', 'L17H5', 'L17H6', 'L17H7', 'L17H8', 'L17H9', 'L17H10', 'L17H11', 'L17H12', 'L17H13', 'L17H14', 'L17H15'], ['L18H0', 'L18H1', 'L18H2', 'L18H3', 'L18H4', 'L18H5', 'L18H6', 'L18H7', 'L18H8', 'L18H9', 'L18H10', 'L18H11', 'L18H12', 'L18H13', 'L18H14', 'L18H15'], ['L19H0', 'L19H1', 'L19H2', 'L19H3', 'L19H4', 'L19H5', 'L19H6', 'L19H7', 'L19H8', 'L19H9', 'L19H10', 'L19H11', 'L19H12', 'L19H13', 'L19H14', 'L19H15'], ['L20H0', 'L20H1', 'L20H2', 'L20H3', 'L20H4', 'L20H5', 'L20H6', 'L20H7', 'L20H8', 'L20H9', 'L20H10', 'L20H11', 'L20H12', 'L20H13', 'L20H14', 'L20H15'], ['L21H0', 'L21H1', 'L21H2', 'L21H3', 'L21H4', 'L21H5', 'L21H6', 'L21H7', 'L21H8', 'L21H9', 'L21H10', 'L21H11', 'L21H12', 'L21H13', 'L21H14', 'L21H15'], ['L22H0', 'L22H1', 'L22H2', 'L22H3', 'L22H4', 'L22H5', 'L22H6', 'L22H7', 'L22H8', 'L22H9', 'L22H10', 'L22H11', 'L22H12', 'L22H13', 'L22H14', 'L22H15'], ['L23H0', 'L23H1', 'L23H2', 'L23H3', 'L23H4', 'L23H5', 'L23H6', 'L23H7', 'L23H8', 'L23H9', 'L23H10', 'L23H11', 'L23H12', 'L23H13', 'L23H14', 'L23H15'], ['0_mlp_out', '1_mlp_out', '2_mlp_out', '3_mlp_out', '4_mlp_out', '5_mlp_out', '6_mlp_out', '7_mlp_out', '8_mlp_out', '9_mlp_out', '10_mlp_out', '11_mlp_out', '12_mlp_out', '13_mlp_out', '14_mlp_out', '15_mlp_out', '16_mlp_out', '17_mlp_out', '18_mlp_out', '19_mlp_out', '20_mlp_out', '21_mlp_out', '22_mlp_out', '23_mlp_out'], ['embed', 'bias']],
        "Llama-3.2-1B-Instruct" : [[f"L{i}H{j}" for j in range(32)] for i in range(16)] + [['0_mlp_out', '1_mlp_out', '2_mlp_out', '3_mlp_out', '4_mlp_out', '5_mlp_out', '6_mlp_out', '7_mlp_out', '8_mlp_out', '9_mlp_out', '10_mlp_out', '11_mlp_out', '12_mlp_out', '13_mlp_out', '14_mlp_out', '15_mlp_out'], ['embed', 'bias']]
    }

    components = []
    for comp_group in x_labels[model_name]:
        for comp in comp_group:
            components.append(comp)

    # DLA
    # Formal
    logit_formal_file = f"scores/logit_attribution/{model_name}/grouped_attr_formal_0_None.pt"
    formal_logits = t.load(logit_formal_file, map_location=t.device(device))
    mean_formal_logits = get_mean_sentence_tensor(formal_logits)
    df_formal_attr = pd.DataFrame(mean_formal_logits.numpy().T, columns=['DLA Idiom Formal', 'DLA Literal Formal'])
    
    std_formal_logits = t.std(formal_logits, dim = 0)
    df_formal_attr["DLA Std Idiom Formal"] = std_formal_logits[0]
    df_formal_attr["DLA Std Literal Formal"] = std_formal_logits[1]
    df_comp = pd.Series(components, name = "Component")

    # Trans
    logit_trans_file = f"scores/logit_attribution/{model_name}/grouped_attr_trans_0_None.pt"
    trans_logits = t.load(logit_trans_file, map_location=t.device(device))
    mean_trans_logits = get_mean_sentence_tensor(trans_logits)
    df_trans_attr = pd.DataFrame(mean_trans_logits.numpy().T, columns=['DLA Idiom Trans', 'DLA Literal Trans'])

    std_trans_logits = t.std(trans_logits, dim = 0)
    df_trans_attr["DLA Std Idiom Trans"] = std_trans_logits[0]
    df_trans_attr["DLA Std Literal Trans"] = std_trans_logits[1]

    # IDIOM SCORE
    # Formal
    idiom_score_formal_file = f"scores/idiom_scores/{model_name}/idiom_formal_0_None.pt"
    formal_idiom_score = t.load(idiom_score_formal_file, map_location=t.device(device))
    mean_formal_idiom_score = get_lh_mean_scores(formal_idiom_score)
    df_formal_idiom_score = pd.DataFrame({"Component": list(mean_formal_idiom_score.keys()), "Idiom Score Formal": list(mean_formal_idiom_score.values())})
    std_formal_idiom_score = get_lh_std_scores(formal_idiom_score)
    df_std_formal_idiom_score = pd.DataFrame({"Component": list(std_formal_idiom_score.keys()), "Idiom Score Std Formal": list(std_formal_idiom_score.values())})
    df_formal_idiom_score = pd.merge(df_formal_idiom_score, df_std_formal_idiom_score, on="Component",  how='left')

    # Trans
    idiom_score_trans_file = f"scores/idiom_scores/{model_name}/idiom_trans_0_None.pt"
    trans_idiom_score = t.load(idiom_score_trans_file, map_location=t.device(device))
    mean_trans_idiom_score = get_lh_mean_scores(trans_idiom_score)
    df_trans_idiom_score = pd.DataFrame({"Component": list(mean_trans_idiom_score.keys()), "Idiom Score Trans": list(mean_trans_idiom_score.values())})
    std_trans_idiom_score = get_lh_std_scores(trans_idiom_score)
    df_std_trans_idiom_score = pd.DataFrame({"Component": list(std_trans_idiom_score.keys()), "Idiom Score Std Trans": list(std_trans_idiom_score.values())})
    df_trans_idiom_score = pd.merge(df_trans_idiom_score, df_std_trans_idiom_score, on="Component",  how='left')

    # LITERAL SCORE
    # Formal
    literal_score_formal_file = f"scores/literal_scores/{model_name}/literal_formal_0_None.pt"
    formal_literal_score = t.load(literal_score_formal_file, map_location=t.device(device))
    mean_formal_literal_score = get_lh_mean_scores(formal_literal_score)
    df_formal_literal_score = pd.DataFrame({"Component": list(mean_formal_literal_score.keys()), "Literal Score Formal": list(mean_formal_literal_score.values())})
    std_formal_literal_score = get_lh_std_scores(formal_literal_score)
    df_std_formal_literal_score = pd.DataFrame({"Component": list(std_formal_literal_score.keys()), "Literal Score Std Formal": list(std_formal_literal_score.values())})
    df_formal_literal_score = pd.merge(df_formal_literal_score, df_std_formal_literal_score, on="Component",  how='left')

    # Trans
    literal_score_trans_file = f"scores/literal_scores/{model_name}/literal_trans_0_None.pt"
    trans_literal_score = t.load(literal_score_trans_file, map_location=t.device(device))
    mean_trans_literal_score = get_lh_mean_scores(trans_literal_score)
    df_trans_literal_score = pd.DataFrame({"Component": list(mean_trans_literal_score.keys()), "Literal Score Trans": list(mean_trans_literal_score.values())})
    std_trans_literal_score = get_lh_std_scores(trans_literal_score)
    df_std_trans_literal_score = pd.DataFrame({"Component": list(std_trans_literal_score.keys()), "Literal Score Std Trans": list(std_trans_literal_score.values())})
    df_trans_literal_score = pd.merge(df_trans_literal_score, df_std_trans_literal_score, on="Component", how='left')

    # ALL
    df = pd.concat([df_comp, df_formal_idiom_score, df_trans_idiom_score, df_formal_literal_score, df_trans_literal_score, df_formal_attr, df_trans_attr], axis=1, join = "outer")
    df["Idiom Score Diff Formal Trans"] = df["Idiom Score Formal"] - df["Idiom Score Trans"]
    df["Literal Score Diff Trans Formal"] = df["Literal Score Trans"] - df["Literal Score Formal"]
    df["Score Diff Idiom Literal Formal"] = df["Idiom Score Formal"] - df["Literal Score Formal"]
    df["DLA Diff Formal"] = df["DLA Idiom Formal"] - df["DLA Literal Formal"]
    df["DLA Diff Trans"] = df["DLA Idiom Trans"] - df["DLA Literal Trans"]
    df["DLA Diff Idiom Formal Trans"] = df["DLA Idiom Formal"] - df["DLA Idiom Trans"]
    df["DLA Diff Literal Formal Trans"] = df["DLA Literal Formal"] - df["DLA Literal Trans"]

    df.to_csv(f"plots/{model_name}/results_{model_name}.csv", index_label = "Index") 

def compute_accuracy(pred_file: str, outfile: str = None):
    """
    This file computes the metrics for the ablation experiments.

    @params
        pred_file: file with the predictions
        outfile: output file with txt-ending
    @returns
        sent_ids: sentences and their ids with the best/worst performance
    """
    with open(pred_file, 'r', encoding = "utf-8") as f:
        predictions = json.load(f)

    num_sents = len(predictions["original_rank"])
    acc = defaultdict(float)
    rank_dif = defaultdict(float) 
    changed_total = defaultdict(float) 
    changed_corr2false = defaultdict(float) 
    changed_false2corr = defaultdict(float) 
    changed_rank_down = defaultdict(float) 
    changed_rank_up = defaultdict(float) 
    wrong = defaultdict(list) 
    worst_pred = defaultdict(str)
    best_pred = defaultdict(str)
    corr2false_pred = defaultdict(str)
    false2corr_pred = defaultdict(str)

    sent_ids = defaultdict(int) 
    for k, val in predictions.items():
        name = k.split('_')[:-1]
        name = '_'.join(name)

        if len(val) != num_sents:
            v = val[:num_sents]
        else:
            v = val

        if "rank" in k:
            acc[name] = t.sum((t.tensor(v) == 0))/num_sents
            rank_diffs = (t.tensor(v)-t.tensor(predictions["original_rank"]))
            rank_dif[name] = t.mean(rank_diffs.float())

            worst_rank = max(rank_diffs)
            best_rank = min(rank_diffs)
            if worst_rank != 0:
                worst_rank_idx = int((rank_diffs == worst_rank).nonzero()[0])
                worst_pred[name] = f"{worst_rank}: {predictions["prompt"][worst_rank_idx]}\n -> '{predictions["correct_token"][worst_rank_idx]}', predicted: '{predictions[name+"_prediction"][worst_rank_idx]}'" 
                sent_ids[predictions["prompt"][worst_rank_idx] + " " + predictions["correct_token"][worst_rank_idx]] =  worst_rank_idx
            else:
                worst_pred[name] = "No rank differences"

            if best_rank != 0:
                best_rank_idx = int((rank_diffs == best_rank).nonzero()[0])
                best_pred[name] = f"{best_rank}: {predictions["prompt"][best_rank_idx]}\n -> '{predictions["correct_token"][best_rank_idx]}', predicted: '{predictions[name+"_prediction"][best_rank_idx]}'" 
                sent_ids[predictions["prompt"][best_rank_idx] + " " + predictions["correct_token"][best_rank_idx]] =  best_rank_idx
            else:
                best_pred[name] = "No rank differences"
        elif "prediction" in k:
            add_sent_id = True
            for i in range(len(v)):
                correct = predictions["correct_token"][i]
                original_tok = predictions["original_prediction"][i]
                original_rank = predictions["original_rank"][i]
                ablated_tok = v[i]
                ablated_rank = predictions[f"{name}_rank"][i]

                if ablated_tok.strip() not in correct.strip():
                    wrong[name].append(f"{predictions["prompt"][i]}\n -> {correct}, predicted: {v[i]}\n")

                if original_rank != ablated_rank:
                    changed_total[name] += 1
                    if original_rank < ablated_rank:
                        changed_rank_down[name] += 1
                    else:
                        changed_rank_up[name] += 1

                    if original_rank == 0:
                        changed_corr2false[name] += 1
                        corr2false_pred[name] = f"Ablated rank {ablated_rank}: {predictions["prompt"][i]}\n -> '{correct}', ablation predicted: '{ablated_tok}'" 
                        
                        if add_sent_id:
                            sent_ids[predictions["prompt"][i] + " " + correct] =  i
                            add_sent_id = False
                    elif original_rank != 0 and ablated_rank == 0:
                        changed_false2corr[name] += 1
                        false2corr_pred[name] = f"Original rank {original_rank}: {predictions["prompt"][i]}\n -> '{correct}', originally predicted: '{original_tok}'" 
                        
                        if add_sent_id:
                            sent_ids[predictions["prompt"][i] + " " + correct] =  i
                            add_sent_id = False

            changed_total[name] = changed_total[name]/num_sents
            changed_corr2false[name] = changed_corr2false[name]/num_sents
            changed_false2corr[name] = changed_false2corr[name]/num_sents
            changed_rank_down[name] = changed_rank_down[name]/num_sents
            changed_rank_up[name] = changed_rank_up[name]/num_sents

    output = ""
    for k, v in acc.items():
        output += f"\n{k}\n"
        output += f"Accuracy: {float(v):.2%}\n"
        output += f"Mean Rank Difference: {float(rank_dif[k])}\n"
        output += f"Total Changed Predictions: {float(changed_total[k]):.2%}\n"
        output += f"Correct2False Changed Predictions: {float(changed_corr2false[k]):.2%}\n"
        output += f"False2Correct Changed Predictions: {float(changed_false2corr[k]):.2%}\n"
        output += f"Ablation has negative effect on rank of the correct token: {float(changed_rank_down[k]):.2%}\n"
        output += f"Ablation has positive effect on rank of the correct token: {float(changed_rank_up[k]):.2%}\n"
        output += f"Worst Prediction: {worst_pred[k]}\n"
        output += f"Best Prediction: {best_pred[k]}\n"
        output += f"Correct2False Prediction: {corr2false_pred[k]}\n"
        output += f"False2Correct Prediction: {false2corr_pred[k]}\n"
        output += f"Wrong Predictions: {wrong[k][:5]}\n"

    if outfile != None:
        with open(outfile, 'w', encoding = "utf-8") as f:
            f.write(output)
    else: 
        print(output)

    return sent_ids

def plot_ablation(logit_file: str, loss_file: str, outfile: str = None, model_name: str = None):
    """
    This function plots the Ablation Scores per experiment. The plot has been improved by Chat-GPT (adding section lines and ordering the heads).

    @params
        logit_file: file with the Ablation Logit Scores
        loss_file: file with the Ablation Loss Scores
        outfile: output file with png-ending
        model_name: model ID
    """
    ablation_heads = {
        "pythia-14m": ["L0H0", "L5H3"],
        "Pythia-1.4B": [[(11, 7)], [(19, 14)], [(13, 4)], [(16, 10)], [(3, 4)], [(18, 4)], [(19, 1)], [(0, 13)], [(15, 13)], [(18, 9)],
                        [(2, 15)], [(14, 5)], [(2, 15), (3, 4), (0, 13)], [(16, 10), (11, 7), (18, 9)],
                        [(19, 14), (19, 1), (13, 4)], [(15, 13), (19, 1), (18, 4)], [(15, 13), (19, 1), (14, 5)],
                        [(2, 15), (16, 10), (19, 14), (15, 13), (15, 13)], [(3, 4), (11, 7), (19, 1), (19, 1), (19, 1)],
                        [(0, 13), (18, 9), (13, 4), (18, 4), (14, 5)]],
        "Llama-3.2-1B-Instruct": [
            [(13, 30)], [(9, 13)], [(15, 8)], [(15, 14)], [(0, 0)], [(12, 30)], [(15, 10)], [(10, 29)], [(0, 21)], [(10, 3)],
            [(15, 12)], [(12, 8)], [(0, 17)],
            [(0, 0), (0, 17), (9, 13)],
            [(12, 8), (10, 29), (15, 12)],
            [(15, 8), (15, 10), (15, 14)],
            [(0, 21), (10, 3), (13, 30)],
            [(10, 3), (12, 30), (13, 30)],
            [(0, 0), (12, 8), (15, 8), (0, 21), (10, 3)],
            [(0, 17), (10, 29), (15, 10), (10, 3), (12, 30)],
            [(9, 13), (15, 12), (15, 14), (13, 30), (13, 30)],
        ],
        "Pythia-1.4B_static": [[(18, 4)], [(19, 14)], [(13, 4)], [(0, 10)], [(3, 4)], [(23, 9)], [(12, 12)], [(17, 2)], [(1, 13)], [(2, 15)], [(3, 4), (2, 15), (0, 10)], [(19, 14), (13, 4), (17, 2)], [(18, 4), (23, 9), (12, 12)], [(19, 14), (1, 13), (23, 9)], [(3, 4), (19, 14), (18, 4), (19, 14)], [(2, 15), (13, 4), (23, 9), (1, 13)], [(0, 10), (17, 2), (12, 12), (23, 9)]],
        "Pythia-1.4B_neg_DLA": [[(10, 5)], [(15, 12)], [(12, 7)], [(12, 7), (15, 12), (10, 5)]]
    }

    custom_order = {
        "Pythia-1.4B": [
            "L2H15", "L3H4", "L0H13", # top 3 idiom score
            "L16H10", "L11H7", "L18H9", # top 3 idiom diff
            "L19H14", "L19H1", "L13H4", # top 3 dla
            "L15H13", "L18H4", # top 3 dla diff formal
            "L14H5", # top 3 dla diff idiom
            "L2H15\nL3H4\nL0H13", "L16H10\nL11H7\nL18H9", "L19H14\nL19H1\nL13H4", "L15H13\nL19H1\nL18H4", "L15H13\nL19H1\nL14H5", # top3 per experiment
            "L2H15\nL16H10\nL19H14\nL15H13\nL15H13", # top 1 all
            "L3H4\nL11H7\nL19H1\nL19H1\nL19H1", # top2 all
            "L0H13\nL18H9\nL13H4\nL18H4\nL14H5" # top 3 all
        ],
        "Llama-3.2-1B-Instruct": [
            "L0H0", "L0H17", "L9H13", "L12H8", "L10H29", "L15H12", "L15H8", "L15H10", "L15H14", "L0H21",
            "L10H3", "L13H30", "L12H30",
            "L0H0\nL0H17\nL9H13", "L12H8\nL10H29\nL15H12", "L15H8\nL15H10\nL15H14", "L0H21\nL10H3\nL13H30",
            "L10H3\nL12H30\nL13H30", "L0H0\nL12H8\nL15H8\nL0H21\nL10H3", "L0H17\nL10H29\nL15H10\nL10H3\nL12H30",
            "L9H13\nL15H12\nL15H14\nL13H30\nL13H30"
        ],
        "Pythia-1.4B_static": [
            "L3H4", "L2H15", "L0H10",
            "L19H14", "L13H4", "L17H2", 
            "L18H4", "L23H9", "L12H12",
            "L1H13",
            "L3H4\nL2H15\nL0H10", "L19H14\nL13H4\nL17H2", "L18H4\nL23H9\nL12H12", "L19H14\nL1H13\nL23H9",
            "L3H4\nL19H14\nL18H4\nL19H14", "L2H15\nL13H4\nL23H9\nL1H13", "L0H10\nL17H2\nL12H12\nL23H9"
        ],
        "Pythia-1.4B_neg_DLA": [
            "L12H7", "L15H12", "L10H5",
            "L12H7\nL15H12\nL10H5"            
        ]   
    }

    separators = {
        "Pythia-1.4B": ["L0H13", "L18H9", "L13H4", "L18H4", "L14H5", "L15H13\nL19H1\nL14H5"],
        "Llama-3.2-1B-Instruct": ["L9H13", "L15H12", "L15H14", "L13H30", "L12H30", "L10H3\nL12H30\nL13H30"],
        "Pythia-1.4B_static": ["L0H10", "L17H2", "L12H12", "L1H13", "L19H14\nL1H13\nL23H9"],
        "Pythia-1.4B_neg_DLA": ["L10H5"]
    }

    if "static" in logit_file:
        section_titles = [
            "Idiom Score", "Idiom Diff", "DLA", "DLA Idiom",
            "Top Per Experiment", "Top All Experiments"
        ]
    elif "neg_DLA" in model_name:
        section_titles = ["Bottom DLA", "Top per Experiment"]
    else:
        section_titles = [
            "Idiom Score", "Idiom Diff", "DLA", "DLA Formal", "DLA Idiom",
            "Top Per Experiment", "Top All Experiments"
        ]

    abl_heads = []
    for group in ablation_heads[model_name]:
        name = ""
        for layer_head in group:
            name += f"\nL{layer_head[0]}H{layer_head[1]}"
        abl_heads.append(name[1:])  # strip first newline


    logit_tensor = t.load(logit_file, map_location=t.device(device))
    loss_tensor = t.load(loss_file, map_location=t.device(device))

    if "neg_DLA" in model_name:
        mean_logit_tensor = get_mean_sentence_tensor(logit_tensor)[:-3]
        mean_loss_tensor = get_mean_sentence_tensor(loss_tensor)[:-3]
    else:
        mean_logit_tensor = get_mean_sentence_tensor(logit_tensor)
        mean_loss_tensor = get_mean_sentence_tensor(loss_tensor)

    df = pd.DataFrame({
        "layer.head": abl_heads,
        "logits": mean_logit_tensor.numpy(),
        "loss": mean_loss_tensor.numpy()
    })

    df["layer.head"] = pd.Categorical(df["layer.head"], categories=custom_order[model_name], ordered=True)
    df = df.sort_values("layer.head").reset_index(drop=True)

    ax = df.plot.bar(x="layer.head", rot=0, fontsize=4, ylim = (-1, 2))
    plt.xlabel("Ablation Group", fontsize=6)
    plt.ylabel("Score", fontsize=6)
    plt.title("Ablation Scores", fontsize=8)

    # Separator lines
    separator_indices = []
    for sep_label in separators[model_name]:
        if sep_label in df["layer.head"].values:
            idx = df[df["layer.head"] == sep_label].index[0]
            separator_indices.append(idx + 0.5)
            plt.axvline(x=idx + 0.5, color='black', linewidth=0.3)

    # Add section titles above each range
    all_split_positions = [0] + [int(s + 0.5) for s in separator_indices] + [len(df)]
    for i in range(len(section_titles)):
        start = all_split_positions[i]
        end = all_split_positions[i + 1]
        center = (start + end - 1) / 2
        plt.text(center, ax.get_ylim()[1] * 0.95, section_titles[i], ha='center', va='bottom', fontsize=4)

    plt.tight_layout()
    plt.legend(prop={'size': 6})
    save_plt(outfile, dpi = 300, bbox_inches='tight')

def plot_logit_diff_per_sent(logit_file, pred_file, outfile = None, model_name = None):
    """
    This function plots the Ablation Logit Scores per sentence. The plot has been improved by Chat-GPT (ordering the sentences).
    """
    ablation_heads = {
        "Pythia-14M": ["L0H0", "L5H3"],
        "Pythia-1.4B": [[(11, 7)], [(19, 14)], [(13, 4)], [(16, 10)], [(3, 4)], [(18, 4)], [(19, 1)], [(0, 13)], [(15, 13)], [(18, 9)], [(2, 15)], [(14, 5)], [(2, 15), (3, 4), (0, 13)], [(16, 10), (11, 7), (18, 9)], [(19, 14), (19, 1), (13, 4)], [(15, 13), (19, 1), (18, 4)], [(15, 13), (19, 1), (14, 5)], [(2, 15), (16, 10), (19, 14), (15, 13), (15, 13)], [(3, 4), (11, 7), (19, 1), (19, 1), (19, 1)], [(0, 13), (18, 9), (13, 4), (18, 4), (14, 5)]], # top heads identified by idiom score and dla
        "Llama-3.2-1B-Instruct": [[(13, 30)], [(9, 13)], [(15, 8)], [(15, 14)], [(0, 0)], [(12, 30)], [(15, 10)], [(10, 29)], [(0, 21)], [(10, 3)], [(15, 12)], [(12, 8)], [(0, 17)], [(0, 0), (0, 17), (9, 13)], [(12, 8), (10, 29), (15, 12)], [(15, 8), (15, 10), (15, 14)], [(0, 21), (10, 3), (13, 30)], [(10, 3), (12, 30), (13, 30)], [(0, 0), (12, 8), (15, 8), (0, 21), (10, 3)], [(0, 17), (10, 29), (15, 10), (10, 3), (12, 30)], [(9, 13), (15, 12), (15, 14), (13, 30), (13, 30)]],
        "Pythia-1.4B_static": [[(18, 4)], [(19, 14)], [(13, 4)], [(0, 10)], [(3, 4)], [(23, 9)], [(12, 12)], [(17, 2)], [(1, 13)], [(2, 15)], [(3, 4), (2, 15), (0, 10)], [(19, 14), (13, 4), (17, 2)], [(18, 4), (23, 9), (12, 12)], [(19, 14), (1, 13), (23, 9)], [(3, 4), (19, 14), (18, 4), (19, 14)], [(2, 15), (13, 4), (23, 9), (1, 13)], [(0, 10), (17, 2), (12, 12), (23, 9)]],
        "Pythia-1.4B_neg_DLA": [[(10, 5)], [(15, 12)], [(12, 7)], [(12, 7), (15, 12), (10, 5)], [(12, 7)], [(15, 12)], [(10, 5)]]
    }

    custom_order = {
        "Pythia-1.4B": [
            "L2H15", "L3H4", "L0H13", # top 3 idiom score
            "L16H10", "L11H7", "L18H9", # top 3 idiom diff
            "L19H14", "L19H1", "L13H4", # top 3 dla
            "L15H13", "L18H4", # top 3 dla diff formal
            "L14H5", # top 3 dla diff idiom
            "L2H15_L3H4_L0H13", "L16H10_L11H7_L18H9", "L19H14_L19H1_L13H4", "L15H13_L19H1_L18H4", "L15H13_L19H1_L14H5", # top3 per experiment
            "L2H15_L16H10_L19H14_L15H13_L15H13", # top 1 all
            "L3H4_L11H7_L19H1_L19H1_L19H1", # top2 all
            "L0H13_L18H9_L13H4_L18H4_L14H5" # top 3 all
        ],
        "Llama-3.2-1B-Instruct": [
            "L0H0", "L0H17", "L9H13", "L12H8", "L10H29", "L15H12", "L15H8", "L15H10", "L15H14", "L0H21",
            "L10H3", "L13H30", "L12H30",
            "L0H0_L0H17_L9H13", "L12H8_L10H29_L15H12", "L15H8_L15H10_L15H14", "L0H21_L10H3_L13H30",
            "L10H3_L12H30_L13H30", 
            "L0H0_L12H8_L15H8_L0H21_L10H3", "L0H17_L10H29_L15H10_L10H3_L12H30",
            "L9H13_L15H12_L15H14_L13H30_L13H30"
        ],
        "Pythia-1.4B_static": [
            "L3H4", "L2H15", "L0H10",
            "L19H14", "L13H4", "L17H2", 
            "L18H4", "L23H9", "L12H12",
            "L1H13", 
            "L3H4_L2H15_L0H10", "L19H14_L13H4_L17H2", "L18H4_L23H9_L12H12", "L19H14_L1H13_L23H9",
            "L3H4_L19H14_L18H4_L19H14", "L2H15_L13H4_L23H9_L1H13", "L0H10_L17H2_L12H12_L23H9"
        ],
        "Pythia-1.4B_neg_DLA": [
            "L12H7", "L15H12", "L10H5",
            "L12H7_L15H12_L10H5"
        ]
    }

    abl_heads = []
    for group in ablation_heads[model_name]:
        name = ""
        for layer_head in group:
            name += f"_L{layer_head[0]}H{layer_head[1]}"
        abl_heads.append(name[1:])

    logit_tensor = t.load(logit_file, map_location=t.device(device))
    logit_tensor = logit_tensor.T

    # {sent: idx}
    sent_ids = compute_accuracy(pred_file, outfile=None)
    sent_ids_tuple = []
    for sent, idx in sent_ids.items():
        if len(sent.split(" ")) > 9:
            sent_ids_tuple.append(("[...] " + " ".join(sent.split(" ")[-9:]), idx))
        else:
            sent_ids_tuple.append((sent, idx))

    sent_ids_tuple = sorted(sent_ids_tuple, key=lambda x: x[0][::-1])
    ids = [x[1] for x in sent_ids_tuple]
    sents = [x[0] for x in sent_ids_tuple] 

    # Create tensor for [sentence x head]
    sent_logit_tensor = t.zeros(len(ids), len(abl_heads))
    for head_pos in range(len(abl_heads)):
        for sent_pos in range(len(ids)):
            sent_logit_tensor[sent_pos][head_pos] = logit_tensor[head_pos][ids[sent_pos]]

    # Reorder according to custom_order
    reorder_indices = [abl_heads.index(name) for name in custom_order[model_name] if name in abl_heads]
    sent_logit_tensor = sent_logit_tensor[:, reorder_indices]
    ordered_abl_heads = [abl_heads[i] for i in reorder_indices]

    # Plot heatmap
    ax = sns.heatmap(sent_logit_tensor, xticklabels=ordered_abl_heads, yticklabels=sents, cbar_kws = {"label": 'Ablation Logit Score'})
    ax.figure.axes[-1].yaxis.label.set_size(6)
    ax.collections[0].colorbar.ax.tick_params(labelsize=6)

    plt.xticks(rotation=90, ha='right', fontsize=4)
    plt.yticks(fontsize=4)
    plt.tight_layout()

    plt.xlabel("Ablation Group", fontsize=6)
    plt.title("Ablation Logit Score", fontsize=8)

    save_plt(outfile, dpi = 300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='idiom head detector')
    parser.add_argument('--model_name', '-m', help='model to run the experiment with', default="Pythia-1.4B")
    parser.add_argument('--tensor_file', '-t', help='file with the tensor scores', default="future_work/scores/idiom_scores/Pythia-1.4B/idiom_static_0_None.pt", type=str)
    parser.add_argument('--image_file', '-i', help='output file for the plot', default=None, type=str)
    parser.add_argument('--scatter_file', '-s', help='file with tensor scores for the scatter plot', default=None, type=str)

    model_name = parser.parse_args().model_name
    tensor_file = parser.parse_args().tensor_file
    img_file = parser.parse_args().image_file
    scatter_file = parser.parse_args().scatter_file
    device = "cuda" if t.cuda.is_available() else "cpu"

    if tensor_file == "None":
        create_csv(model_name, device)
    else:
        if tensor_file.endswith(".json"):
            # Ablation
            if img_file != None:
                if "static" in img_file:
                    os.makedirs(f"./future_work/plots/{model_name}/ablation", exist_ok=True)
                    txt_file = f"./future_work/plots/{model_name}/ablation/{img_file}.txt"
                    png_file = f"./future_work/plots/{model_name}/ablation/{img_file}.png"
                    heat_file = f"./future_work/plots/{model_name}/ablation/heat_{img_file}.png"
                else:
                    os.makedirs(f"./plots/{model_name}/ablation", exist_ok=True)
                    txt_file = f"./plots/{model_name}/ablation/{img_file}.txt"
                    png_file = f"./plots/{model_name}/ablation/{img_file}.png"
                    heat_file = f"./plots/{model_name}/ablation/heat_{img_file}.png"
            else:
                filename = None
                heat_file = None
                txt_file = None
                png_file = None

            compute_accuracy(tensor_file, txt_file)

            splits = tensor_file.split('_')
            split = ""
            for part in splits:
                if part in ["formal", "trans", "static"]:
                    split = part

            if "static" in tensor_file:
                logit_file = f"future_work/scores/ablation/Pythia-1.4B_static/ablation_static_0_2761_logit.pt"
                loss_file = f"future_work/scores/ablation/Pythia-1.4B_static/ablation_static_0_2761_loss.pt"
                model_name = "Pythia-1.4B_static"
            else:
                logit_file = f"scores/ablation/{model_name}/ablation_{split}_0_None_logit.pt"
                loss_file = f"scores/ablation/{model_name}/ablation_{split}_0_None_loss.pt"

            plot_logit_diff_per_sent(logit_file, tensor_file, outfile=heat_file, model_name=model_name)
            plot_ablation(logit_file, loss_file, png_file, model_name)
        else:
            loaded_tensor = t.load(tensor_file, map_location=t.device(device))
            print(f"Loaded tensor with size: {loaded_tensor.size()}")

            if tensor_file.endswith("_comp.pt"):
                if "idiom" in tensor_file:
                    path = f"./plots/{model_name}/idiom_components"

                    if "static" in tensor_file:
                        path = f"./future_work/plots/{model_name}/idiom_components"
                elif "literal" in tensor_file:
                    path = f"./plots/{model_name}/literal_components"

                    if "static" in tensor_file:
                        path = f"./future_work/plots/{model_name}/literal_components"
                os.makedirs(path, exist_ok=True)

                plot_all_components(loaded_tensor, img_file, model_name, path = path)
            elif "grouped" in tensor_file:
                if "contr" in tensor_file:
                    path = f"./plots/{model_name}/contribution"
                    os.makedirs(path, exist_ok=True)
                    x_labels = {
                        "Llama-3.2-1B-Instruct" : [[f"L{i}H{j}" for j in range(32)] for i in range(16)]
                    }
                    # layer x group x head 
                    mean_tensor = get_mean_sentence_tensor(loaded_tensor)

                    for comp_group in range(len(x_labels[model_name])):
                        num_comps = len(x_labels[model_name][comp_group])
                        comp_contr = mean_tensor[comp_group, :, :]
                        plot_logit_attribution_split(comp_contr, x_labels = x_labels[model_name][comp_group], filename=f"{path}/{img_file}_{comp_group}.png")
                else: 
                    if img_file != None:
                        path = f"./plots/{model_name}/logit_attribution"
                        os.makedirs(path, exist_ok=True)

                    x_labels = {
                        "pythia-14m": ["L0H0", "L0H1", "L0H2", "L0H3", "L1H0", "L1H1", "L1H2", "L1H3", "L2H0", "L2H1", "L2H2", "L2H3", "L3H0", "L3H1", "L3H2", "L3H3", "L4H0", "L4H1", "L4H2", "L4H3", "L5H0", "L5H1", "L5H2", "L5H3", "0_mlp_out", "1_mlp_out", "2_mlp_out", "3_mlp_out", "4_mlp_out", "5_mlp_out", "embed", "bias"],
                        "pythia-1.4b": [['L0H0', 'L0H1', 'L0H2', 'L0H3', 'L0H4', 'L0H5', 'L0H6', 'L0H7', 'L0H8', 'L0H9', 'L0H10', 'L0H11', 'L0H12', 'L0H13', 'L0H14', 'L0H15'], ['L1H0', 'L1H1', 'L1H2', 'L1H3', 'L1H4', 'L1H5', 'L1H6', 'L1H7', 'L1H8', 'L1H9', 'L1H10', 'L1H11', 'L1H12', 'L1H13', 'L1H14', 'L1H15'], ['L2H0', 'L2H1', 'L2H2', 'L2H3', 'L2H4', 'L2H5', 'L2H6', 'L2H7', 'L2H8', 'L2H9', 'L2H10', 'L2H11', 'L2H12', 'L2H13', 'L2H14', 'L2H15'], ['L3H0', 'L3H1', 'L3H2', 'L3H3', 'L3H4', 'L3H5', 'L3H6', 'L3H7', 'L3H8', 'L3H9', 'L3H10', 'L3H11', 'L3H12', 'L3H13', 'L3H14', 'L3H15'], ['L4H0', 'L4H1', 'L4H2', 'L4H3', 'L4H4', 'L4H5', 'L4H6', 'L4H7', 'L4H8', 'L4H9', 'L4H10', 'L4H11', 'L4H12', 'L4H13', 'L4H14', 'L4H15'], ['L5H0', 'L5H1', 'L5H2', 'L5H3', 'L5H4', 'L5H5', 'L5H6', 'L5H7', 'L5H8', 'L5H9', 'L5H10', 'L5H11', 'L5H12', 'L5H13', 'L5H14', 'L5H15'], ['L6H0', 'L6H1', 'L6H2', 'L6H3', 'L6H4', 'L6H5', 'L6H6', 'L6H7', 'L6H8', 'L6H9', 'L6H10', 'L6H11', 'L6H12', 'L6H13', 'L6H14', 'L6H15'], ['L7H0', 'L7H1', 'L7H2', 'L7H3', 'L7H4', 'L7H5', 'L7H6', 'L7H7', 'L7H8', 'L7H9', 'L7H10', 'L7H11', 'L7H12', 'L7H13', 'L7H14', 'L7H15'], ['L8H0', 'L8H1', 'L8H2', 'L8H3', 'L8H4', 'L8H5', 'L8H6', 'L8H7', 'L8H8', 'L8H9', 'L8H10', 'L8H11', 'L8H12', 'L8H13', 'L8H14', 'L8H15'], ['L9H0', 'L9H1', 'L9H2', 'L9H3', 'L9H4', 'L9H5', 'L9H6', 'L9H7', 'L9H8', 'L9H9', 'L9H10', 'L9H11', 'L9H12', 'L9H13', 'L9H14', 'L9H15'], ['L10H0', 'L10H1', 'L10H2', 'L10H3', 'L10H4', 'L10H5', 'L10H6', 'L10H7', 'L10H8', 'L10H9', 'L10H10', 'L10H11', 'L10H12', 'L10H13', 'L10H14', 'L10H15'], ['L11H0', 'L11H1', 'L11H2', 'L11H3', 'L11H4', 'L11H5', 'L11H6', 'L11H7', 'L11H8', 'L11H9', 'L11H10', 'L11H11', 'L11H12', 'L11H13', 'L11H14', 'L11H15'], ['L12H0', 'L12H1', 'L12H2', 'L12H3', 'L12H4', 'L12H5', 'L12H6', 'L12H7', 'L12H8', 'L12H9', 'L12H10', 'L12H11', 'L12H12', 'L12H13', 'L12H14', 'L12H15'], ['L13H0', 'L13H1', 'L13H2', 'L13H3', 'L13H4', 'L13H5', 'L13H6', 'L13H7', 'L13H8', 'L13H9', 'L13H10', 'L13H11', 'L13H12', 'L13H13', 'L13H14', 'L13H15'], ['L14H0', 'L14H1', 'L14H2', 'L14H3', 'L14H4', 'L14H5', 'L14H6', 'L14H7', 'L14H8', 'L14H9', 'L14H10', 'L14H11', 'L14H12', 'L14H13', 'L14H14', 'L14H15'], ['L15H0', 'L15H1', 'L15H2', 'L15H3', 'L15H4', 'L15H5', 'L15H6', 'L15H7', 'L15H8', 'L15H9', 'L15H10', 'L15H11', 'L15H12', 'L15H13', 'L15H14', 'L15H15'], ['L16H0', 'L16H1', 'L16H2', 'L16H3', 'L16H4', 'L16H5', 'L16H6', 'L16H7', 'L16H8', 'L16H9', 'L16H10', 'L16H11', 'L16H12', 'L16H13', 'L16H14', 'L16H15'], ['L17H0', 'L17H1', 'L17H2', 'L17H3', 'L17H4', 'L17H5', 'L17H6', 'L17H7', 'L17H8', 'L17H9', 'L17H10', 'L17H11', 'L17H12', 'L17H13', 'L17H14', 'L17H15'], ['L18H0', 'L18H1', 'L18H2', 'L18H3', 'L18H4', 'L18H5', 'L18H6', 'L18H7', 'L18H8', 'L18H9', 'L18H10', 'L18H11', 'L18H12', 'L18H13', 'L18H14', 'L18H15'], ['L19H0', 'L19H1', 'L19H2', 'L19H3', 'L19H4', 'L19H5', 'L19H6', 'L19H7', 'L19H8', 'L19H9', 'L19H10', 'L19H11', 'L19H12', 'L19H13', 'L19H14', 'L19H15'], ['L20H0', 'L20H1', 'L20H2', 'L20H3', 'L20H4', 'L20H5', 'L20H6', 'L20H7', 'L20H8', 'L20H9', 'L20H10', 'L20H11', 'L20H12', 'L20H13', 'L20H14', 'L20H15'], ['L21H0', 'L21H1', 'L21H2', 'L21H3', 'L21H4', 'L21H5', 'L21H6', 'L21H7', 'L21H8', 'L21H9', 'L21H10', 'L21H11', 'L21H12', 'L21H13', 'L21H14', 'L21H15'], ['L22H0', 'L22H1', 'L22H2', 'L22H3', 'L22H4', 'L22H5', 'L22H6', 'L22H7', 'L22H8', 'L22H9', 'L22H10', 'L22H11', 'L22H12', 'L22H13', 'L22H14', 'L22H15'], ['L23H0', 'L23H1', 'L23H2', 'L23H3', 'L23H4', 'L23H5', 'L23H6', 'L23H7', 'L23H8', 'L23H9', 'L23H10', 'L23H11', 'L23H12', 'L23H13', 'L23H14', 'L23H15'], ['0_mlp_out', '1_mlp_out', '2_mlp_out', '3_mlp_out', '4_mlp_out', '5_mlp_out', '6_mlp_out', '7_mlp_out', '8_mlp_out', '9_mlp_out', '10_mlp_out', '11_mlp_out', '12_mlp_out', '13_mlp_out', '14_mlp_out', '15_mlp_out', '16_mlp_out', '17_mlp_out', '18_mlp_out', '19_mlp_out', '20_mlp_out', '21_mlp_out', '22_mlp_out', '23_mlp_out'], ['embed', 'bias']],
                        "Llama-3.2-1B-Instruct" : [[f"L{i}H{j}" for j in range(32)] for i in range(16)] + [['0_mlp_out', '1_mlp_out', '2_mlp_out', '3_mlp_out', '4_mlp_out', '5_mlp_out', '6_mlp_out', '7_mlp_out', '8_mlp_out', '9_mlp_out', '10_mlp_out', '11_mlp_out', '12_mlp_out', '13_mlp_out', '14_mlp_out', '15_mlp_out'], ['embed', 'bias']]
                    }

                    mean_tensor = get_mean_sentence_tensor(loaded_tensor)
                    seen_comps = 0
                    for comp_group in range(len(x_labels[model_name])):
                        layer = ""
                        first_comp = x_labels[model_name][comp_group][0]
                        if first_comp.startswith('L'):
                            layer = f"Layer {first_comp.split("H")[0][1:]}"
                        elif first_comp == "embed":
                            layer = "Embed/Bias"
                        else:
                            layer = "MLP"

                        if img_file != None:
                            filename = f"{path}/{img_file}_{comp_group}.png"
                        else:
                            filename = None

                        num_comps = len(x_labels[model_name][comp_group])
                        comp_logit = mean_tensor[:, seen_comps:(seen_comps+num_comps)]
                        plot_logit_attribution_split(comp_logit, x_labels = x_labels[model_name][comp_group], filename=filename, title = f"Direct Logit Attribution {layer}")
                        seen_comps += num_comps
            elif "loss" in tensor_file:
                plot_loss(loaded_tensor, img_file, model_name)
            else:
                plot_idiom_scores(loaded_tensor, img_file, model_name, scatter_file)


    
