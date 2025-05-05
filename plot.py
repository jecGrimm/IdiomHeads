from IPython.display import display
import circuitsvis as cv
import pandas as pd
import matplotlib.pyplot as plt
import torch as t
import plotly.express as px
import argparse
import os
import numpy as np
from collections import defaultdict
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from transformer_lens import utils
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from helper import get_logit_component

# %pip install jaxtyping
# %pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python

def plot_attention_heads(model, cache, str_tokens):
    for layer in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", layer]
        display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern, attention_head_names=[f"L{layer}H{i}" for i in range(model.cfg.n_heads)]))

        del attention_pattern
        t.cuda.empty_cache()

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

def plot_tensor_line(tensor, filename = None, title = "Average score per head and layer", ylabel = "Mean score"):
    mean_tensor = get_mean_sentence_tensor(tensor)
    df = create_df_from_tensor(mean_tensor)
    df.plot.line(title= title, xlabel = "Layer", ylabel = ylabel, xticks = np.arange(mean_tensor.size(0)), yticks = np.arange(-1.0, 1.01, 0.1), colormap = "tab20", figsize=(25, 25))
    plt.legend(title = "Head")
    save_plt(filename)

def plot_line_std(tensor, filename = None, title = "Standard deviation of the score per head and layer", ylabel = "Standard deviation of the score"):
    std_tensor = t.std(tensor, dim=0)
    df = create_df_from_tensor(std_tensor)
    #print("std:", std_tensor.size())
    df.plot.line(title=title, xlabel = "Layer", ylabel = ylabel, xticks = np.arange(std_tensor.size(0)), colormap = "tab20", figsize=(25, 25))
    plt.legend(title = "Head")
    save_plt(filename)

def plot_tensor_hist(tensor, filename = None, title = "Distribution of the mean scores", xlabel = "Mean scores"):
    mean_tensor = get_mean_sentence_tensor(tensor)
    len = mean_tensor.size(0) * mean_tensor.size(1)
    df = create_df_from_tensor(mean_tensor.view(len))
    df.plot.hist(title=title, legend = False, xlabel=xlabel, yticks = range(0, 120, 10))
    
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

def plot_heatmap(tensor, filename = None, title = "Scores"):
    mean_tensor = get_mean_sentence_tensor(tensor)
    #print(mean_tensor[0][0])
    fig = px.imshow(mean_tensor, labels=dict(x="Head", y="Layer"), title = title)

    if filename:
        fig.write_image(filename)
    else:
        fig.show()

def plot_scatter(idiom_tensor, literal_tensor, filename = None):
    idiom_mean = get_lh_mean_scores(idiom_tensor) # dict layer.head = mean over all sentences
    literal_mean = get_lh_mean_scores(literal_tensor) 

    #df = pd.DataFrame({"layer.head": list(idiom_mean.keys()), "idiom_mean": list(idiom_mean.values()), "literal_mean": list(literal_mean.values())}).set_index("layer.head")
    df = pd.DataFrame({"layer.head": list(idiom_mean.keys()), "idiom_mean": list(idiom_mean.values()), "literal_mean": list(literal_mean.values())})
    df.plot.scatter(x='idiom_mean',y='literal_mean')

    save_plt(filename)

def plot_scatter_components(comp_dict, filename = None):
    df = pd.DataFrame()
    #print("\ncomps:\n", comp_dict.keys())
    for comp, tensor in comp_dict.items():
        comp_lh = get_lh_mean_scores(tensor)
        #print("\ncomp_lh\n", comp_lh)
        comp_df = pd.DataFrame({"layer.head": list(comp_lh.keys()), comp: list(comp_lh.values())}).set_index("layer.head")
        #print("\ncomp_df\n", comp_df)
        if df.empty:
            df = comp_df
        else:
            df = df.join(comp_df, on='layer.head')
    
    #print("\ndf\n", df)
    #df = pd.DataFrame({"layer.head": list(idiom_mean.keys()), "idiom_mean": list(idiom_mean.values()), "literal_mean": list(literal_mean.values())}).set_index("layer.head")
    #df = pd.DataFrame({"layer.head": list(idiom_mean.keys()), "idiom_mean": list(idiom_mean.values()), "literal_mean": list(literal_mean.values())})
    #fig, axes = plt.subplots(nrows = 5, ncols = 5, constrained_layout=True, sharex=True, sharey=True, figsize=(15,15))
    fig, axes = plt.subplots(nrows = 5, ncols = 5, constrained_layout=True, figsize=(20,20))
    #plt.subplots_adjust(wspace=0.5, hspace=0.5)
    #df.plot.scatter(x='Mean',y='Std', subplots=True, title = "Scatter plot of the components")
    comps = list(comp_dict.keys())
    for i in range(5):
        for j in range(5):
            axes[i,j].set_title(f"{comps[i]}/{comps[j]}")
            axes[i,j].scatter(df[comps[i]],df[comps[j]])

            # confidence_ellipse(df[comps[i]],df[comps[j]], axes[i, j], edgecolor='red')
            # axes[i,j].scatter(0, 0, c='red', s=3)
    save_plt(filename)

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    VON MATPLOTLIB DEMO https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

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
            scores[f"L{layer}H{head}"] = t.mean(sent_last[layer][head]).numpy()
    return scores

def get_lh_std_scores(tensor):
    scores = dict()
    sent_last = t.einsum("ijk->jki", tensor)
    for layer in range(tensor.size(1)):
        for head in range(tensor.size(2)):
            scores[f"L{layer}H{head}"] = t.std(sent_last[layer][head]).numpy()
    return scores

def explore_scores(tensor, filename = None):
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

    mean_top_10_layer, mean_top_10_heads = get_dist(mean_top_10)
    output += f"\nMean distribution top 10 layer: {mean_top_10_layer}"
    output += f"\nMean distribution top 10 heads: {mean_top_10_heads}"

    mean_bottom_10_layer, mean_bottom_10_heads = get_dist(mean_bottom_10)
    output += f"\nMean distribution bottom 10 layer: {mean_bottom_10_layer}"
    output += f"\nMean distribution bottom 10 heads: {mean_bottom_10_heads}"

    # mean_above_59 = {head: float(score) for head, score in mean_scores.items() if score >= 0.59}
    # output += f"\nThere are {len(mean_above_59)} scores with a score higher than 0.59: {sorted(mean_above_59, key = lambda k:mean_above_59.get(k), reverse = True)}"
    # STD
    std_scores = get_lh_std_scores(tensor)
    # std_above_59 = [(head, float(std_scores[head])) for head in mean_above_59.keys()]
    # output += f"\nThese scores have a std of: {std_above_59}"
    output += f"\n\nMaximum std: {max(std_scores, key=lambda k:std_scores.get(k))} - {max(std_scores.values())}"
    output += f"\nMinimum std: {min(std_scores, key=lambda k:std_scores.get(k))} - {min(std_scores.values())}"

    std_top_10 = [(head, std_scores.get(head)) for head in sorted(std_scores, key = lambda k:std_scores.get(k), reverse = True)][:10] 
    std_bottom_10 = [(head, std_scores.get(head)) for head in sorted(std_scores, key = lambda k:std_scores.get(k))][:10]
    output += f"\nStd top 10: {std_top_10}"
    output += f"\nStd bottom 10: {std_bottom_10}"

    std_top_10_layer, std_top_10_heads = get_dist(std_top_10)
    output += f"\nStd distribution top 10 layer: {std_top_10_layer}"
    output += f"\nStd distribution top 10 heads: {std_top_10_heads}"

    std_bottom_10_layer, std_bottom_10_heads = get_dist(std_bottom_10)
    output += f"\nStd distribution bottom 10 layer: {std_bottom_10_layer}"
    output += f"\nStd distribution bottom 10 heads: {std_bottom_10_heads}"

    if filename != None:
        with open(filename, 'w', encoding = "utf-8") as f:
            f.write(output)
    else:
        print(output)

def get_dist(score_list):
    layer_count = defaultdict(int)
    head_count = defaultdict(int)
    for layer_head, _ in score_list:
        layer, head = layer_head.split(".")
        layer = int(layer)
        head = int(head)
        if layer < 6:
            layer_count["first"] += 1
        elif layer >= 6 and layer < 12:
            layer_count["second"] += 1
        elif layer >= 12 and  layer < 18:
            layer_count["third"] += 1
        else:
            layer_count["fourth"] += 1

        if head < 4:
            head_count["first"] += 1
        elif head >= 4 and layer < 8:
            head_count["second"] += 1
        elif head >= 8 and  layer < 12:
            head_count["third"] += 1
        else:
            head_count["fourth"] += 1
    return layer_count, head_count

def get_head_info(layer_head, tensor):
    mean_scores = get_lh_mean_scores(tensor)
    ranked_mean = list(create_df_from_dict(mean_scores).sort_values(by="scores", ascending = False).index)
    std_scores = get_lh_std_scores(tensor)

    print(f"\nHead: {layer_head}\n\tScore: {mean_scores[layer_head]}\n\tStd: {std_scores[layer_head]}\n\tRank: {ranked_mean.index(layer_head)+1}")

def get_logit_info(num, tensor, model_name):
    mean_logit_attr = get_mean_sentence_tensor(tensor)
    print(f"\nNumber {num}, Component {get_logit_component(num, model_name)}\nIdiom Logit: {mean_logit_attr[num][0]}\nLiteral Logit: {mean_logit_attr[num][1]}")

def imshow(tensor, xaxis="", yaxis="", caxis="", **kwargs):
    return px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)

def convert_tokens_to_string(model, tokens, batch_index=0):
    '''
    Helper function to convert tokens into a list of strings, for printing.
    '''
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{model.tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]

def plot_logit_attribution_sentence(logit_attr: t.Tensor, tokens: t.Tensor, title: str = "", xlabels = None):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(tokens)
    return imshow(logit_attr.float(), x=xlabels,  y=y_labels, xaxis="Component", yaxis="Position", caxis="logit", title=title if title else None, height=25*len(tokens))

def plot_logit_attribution_split(logit_attr: t.Tensor, title: str = "", x_labels = None, filename = None):
    #fig = make_subplots(len(x_labels), 1)
    #mean_logit_attr = get_mean_sentence_tensor(logit_attr)
    y_labels = ["Idiom", "Literal"]

    # for comp_group in range(len(x_labels)):
    #     num_comps = len(x_labels[comp_group])

    #     comp_logit = mean_logit_attr[:, comp_group*num_comps:(comp_group*num_comps+num_comps)]

    #     fig.add_trace(go.Heatmap(z=comp_logit.float()), row=comp_group+1, col=1)
        #fig = fig.add_heatmap(y=[0, 1], x = x_labels[comp_group], z = , row=comp_group, col=1)
        #fig = fig.add_trace(px.imshow(comp_logit.float(), title = title, x=x_labels[comp_group], y=y_labels, labels={"x":"Component", "y":"Position", "color":"Mean logit attribution"}, color_continuous_midpoint=0.0))
    fig = px.imshow(logit_attr.float(), title = title, x=x_labels, y=y_labels, labels={"x":"Component", "y":"Position", "color":"Mean logit attribution"}, color_continuous_midpoint=0.0)
    #fig.update_yaxes(dict(anchor="free", automargin=True, autoshift=True))
    if filename:
        fig.write_image(filename)
    else:
        fig.show()

def plot_all(tensor, filename = None, model_name = None, scatter_file = None):
    path = f"./plots/{model_name}"
    if filename and model_name:
        plot_tensor_line(tensor, f"{path}/mean_line_{filename}.png")
        plot_line_std(tensor, f"{path}/std_line_{filename}.png")
        plot_heatmap(tensor, f"{path}/heat_{filename}.png")
        plot_tensor_hist(tensor, f"{path}/hist_{filename}.png")
        explore_scores(tensor, f"{path}/{filename}.txt")
    else:
        plot_tensor_line(tensor)
        plot_line_std(tensor)
        plot_heatmap(tensor)
        plot_tensor_hist(tensor)
        explore_scores(tensor)

    if scatter_file != None:
        device = "cuda" if t.cuda.is_available() else "cpu"
        scatter_tensor = t.load(scatter_file, map_location=t.device(device))
        print(f"Loaded tensor with size: {scatter_tensor.size()}")

        if filename and model_name:
            plot_scatter(tensor, scatter_tensor, f"plots/{model_name}/scatter_{filename}_literal.png")
        else:
            plot_scatter(tensor, scatter_tensor)
    
def get_component_dict(tensor):
    comp_dict = {
        "Mean": tensor[:, :, :, 0],
        "Std": tensor[:, :, :, 1],
        "Max": tensor[:, :, :, 2],
        "Phrase": tensor[:, :, :, 3],
        "Contribution": tensor[:, :, :, 4] 
    }
    return comp_dict

def plot_all_components(full_tensor, filename = None, model_name = None):
    path = f"./plots/{model_name}"

    comp_dict = get_component_dict(full_tensor)
    orig_filename = filename
    for comp, tensor in comp_dict.items():
        if orig_filename and model_name:
            filename = orig_filename + f"_{comp}"
            plot_tensor_line(tensor, f"{path}/mean_line_{filename}.png", title = f"Average {comp} (component) per head and layer", ylabel=comp)
            plot_line_std(tensor, f"{path}/std_line_{filename}.png", title = f"Standard deviation of the {comp} (component) per head and layer", ylabel = comp)
            plot_heatmap(tensor, f"{path}/heat_{filename}.png", title = f"{comp[0].upper()}{comp[1:]} (component)")
            plot_tensor_hist(tensor, f"{path}/hist_{filename}.png", title = f"Distribution of the {comp} (component)", xlabel = comp)
            explore_scores(tensor, f"{path}/{filename}.txt")

        else:
            plot_tensor_line(tensor, title = f"Average {comp} (component) per head and layer", ylabel=comp)
            plot_line_std(tensor, title = f"Standard deviation of the {comp} (component) per head and layer", ylabel=comp)
            plot_heatmap(tensor, title = f"{comp[0].upper()}{comp[1:]} (component)")
            plot_tensor_hist(tensor, title = f"Distribution of the {comp} (component)", xlabel=comp)
            print(f"\n\nComponent: {comp}")
            explore_scores(tensor)

    if orig_filename and model_name:
        plot_scatter_components(comp_dict, f"{path}/scatter_{orig_filename}_comp.png")
    else:
        plot_scatter_components(comp_dict)

def create_csv(model_name, device):
    x_labels = {
        "pythia-14m": [["L0H0", "L0H1", "L0H2", "L0H3", "L1H0", "L1H1", "L1H2", "L1H3", "L2H0", "L2H1", "L2H2", "L2H3", "L3H0", "L3H1", "L3H2", "L3H3", "L4H0", "L4H1", "L4H2", "L4H3", "L5H0", "L5H1", "L5H2", "L5H3", "0_mlp_out", "1_mlp_out", "2_mlp_out", "3_mlp_out", "4_mlp_out", "5_mlp_out", "embed", "bias"]],
        "pythia-1.4b": [['L0H0', 'L0H1', 'L0H2', 'L0H3', 'L0H4', 'L0H5', 'L0H6', 'L0H7', 'L0H8', 'L0H9', 'L0H10', 'L0H11', 'L0H12', 'L0H13', 'L0H14', 'L0H15'], ['L1H0', 'L1H1', 'L1H2', 'L1H3', 'L1H4', 'L1H5', 'L1H6', 'L1H7', 'L1H8', 'L1H9', 'L1H10', 'L1H11', 'L1H12', 'L1H13', 'L1H14', 'L1H15'], ['L2H0', 'L2H1', 'L2H2', 'L2H3', 'L2H4', 'L2H5', 'L2H6', 'L2H7', 'L2H8', 'L2H9', 'L2H10', 'L2H11', 'L2H12', 'L2H13', 'L2H14', 'L2H15'], ['L3H0', 'L3H1', 'L3H2', 'L3H3', 'L3H4', 'L3H5', 'L3H6', 'L3H7', 'L3H8', 'L3H9', 'L3H10', 'L3H11', 'L3H12', 'L3H13', 'L3H14', 'L3H15'], ['L4H0', 'L4H1', 'L4H2', 'L4H3', 'L4H4', 'L4H5', 'L4H6', 'L4H7', 'L4H8', 'L4H9', 'L4H10', 'L4H11', 'L4H12', 'L4H13', 'L4H14', 'L4H15'], ['L5H0', 'L5H1', 'L5H2', 'L5H3', 'L5H4', 'L5H5', 'L5H6', 'L5H7', 'L5H8', 'L5H9', 'L5H10', 'L5H11', 'L5H12', 'L5H13', 'L5H14', 'L5H15'], ['L6H0', 'L6H1', 'L6H2', 'L6H3', 'L6H4', 'L6H5', 'L6H6', 'L6H7', 'L6H8', 'L6H9', 'L6H10', 'L6H11', 'L6H12', 'L6H13', 'L6H14', 'L6H15'], ['L7H0', 'L7H1', 'L7H2', 'L7H3', 'L7H4', 'L7H5', 'L7H6', 'L7H7', 'L7H8', 'L7H9', 'L7H10', 'L7H11', 'L7H12', 'L7H13', 'L7H14', 'L7H15'], ['L8H0', 'L8H1', 'L8H2', 'L8H3', 'L8H4', 'L8H5', 'L8H6', 'L8H7', 'L8H8', 'L8H9', 'L8H10', 'L8H11', 'L8H12', 'L8H13', 'L8H14', 'L8H15'], ['L9H0', 'L9H1', 'L9H2', 'L9H3', 'L9H4', 'L9H5', 'L9H6', 'L9H7', 'L9H8', 'L9H9', 'L9H10', 'L9H11', 'L9H12', 'L9H13', 'L9H14', 'L9H15'], ['L10H0', 'L10H1', 'L10H2', 'L10H3', 'L10H4', 'L10H5', 'L10H6', 'L10H7', 'L10H8', 'L10H9', 'L10H10', 'L10H11', 'L10H12', 'L10H13', 'L10H14', 'L10H15'], ['L11H0', 'L11H1', 'L11H2', 'L11H3', 'L11H4', 'L11H5', 'L11H6', 'L11H7', 'L11H8', 'L11H9', 'L11H10', 'L11H11', 'L11H12', 'L11H13', 'L11H14', 'L11H15'], ['L12H0', 'L12H1', 'L12H2', 'L12H3', 'L12H4', 'L12H5', 'L12H6', 'L12H7', 'L12H8', 'L12H9', 'L12H10', 'L12H11', 'L12H12', 'L12H13', 'L12H14', 'L12H15'], ['L13H0', 'L13H1', 'L13H2', 'L13H3', 'L13H4', 'L13H5', 'L13H6', 'L13H7', 'L13H8', 'L13H9', 'L13H10', 'L13H11', 'L13H12', 'L13H13', 'L13H14', 'L13H15'], ['L14H0', 'L14H1', 'L14H2', 'L14H3', 'L14H4', 'L14H5', 'L14H6', 'L14H7', 'L14H8', 'L14H9', 'L14H10', 'L14H11', 'L14H12', 'L14H13', 'L14H14', 'L14H15'], ['L15H0', 'L15H1', 'L15H2', 'L15H3', 'L15H4', 'L15H5', 'L15H6', 'L15H7', 'L15H8', 'L15H9', 'L15H10', 'L15H11', 'L15H12', 'L15H13', 'L15H14', 'L15H15'], ['L16H0', 'L16H1', 'L16H2', 'L16H3', 'L16H4', 'L16H5', 'L16H6', 'L16H7', 'L16H8', 'L16H9', 'L16H10', 'L16H11', 'L16H12', 'L16H13', 'L16H14', 'L16H15'], ['L17H0', 'L17H1', 'L17H2', 'L17H3', 'L17H4', 'L17H5', 'L17H6', 'L17H7', 'L17H8', 'L17H9', 'L17H10', 'L17H11', 'L17H12', 'L17H13', 'L17H14', 'L17H15'], ['L18H0', 'L18H1', 'L18H2', 'L18H3', 'L18H4', 'L18H5', 'L18H6', 'L18H7', 'L18H8', 'L18H9', 'L18H10', 'L18H11', 'L18H12', 'L18H13', 'L18H14', 'L18H15'], ['L19H0', 'L19H1', 'L19H2', 'L19H3', 'L19H4', 'L19H5', 'L19H6', 'L19H7', 'L19H8', 'L19H9', 'L19H10', 'L19H11', 'L19H12', 'L19H13', 'L19H14', 'L19H15'], ['L20H0', 'L20H1', 'L20H2', 'L20H3', 'L20H4', 'L20H5', 'L20H6', 'L20H7', 'L20H8', 'L20H9', 'L20H10', 'L20H11', 'L20H12', 'L20H13', 'L20H14', 'L20H15'], ['L21H0', 'L21H1', 'L21H2', 'L21H3', 'L21H4', 'L21H5', 'L21H6', 'L21H7', 'L21H8', 'L21H9', 'L21H10', 'L21H11', 'L21H12', 'L21H13', 'L21H14', 'L21H15'], ['L22H0', 'L22H1', 'L22H2', 'L22H3', 'L22H4', 'L22H5', 'L22H6', 'L22H7', 'L22H8', 'L22H9', 'L22H10', 'L22H11', 'L22H12', 'L22H13', 'L22H14', 'L22H15'], ['L23H0', 'L23H1', 'L23H2', 'L23H3', 'L23H4', 'L23H5', 'L23H6', 'L23H7', 'L23H8', 'L23H9', 'L23H10', 'L23H11', 'L23H12', 'L23H13', 'L23H14', 'L23H15'], ['0_mlp_out', '1_mlp_out', '2_mlp_out', '3_mlp_out', '4_mlp_out', '5_mlp_out', '6_mlp_out', '7_mlp_out', '8_mlp_out', '9_mlp_out', '10_mlp_out', '11_mlp_out', '12_mlp_out', '13_mlp_out', '14_mlp_out', '15_mlp_out', '16_mlp_out', '17_mlp_out', '18_mlp_out', '19_mlp_out', '20_mlp_out', '21_mlp_out', '22_mlp_out', '23_mlp_out'], ['embed', 'bias']]
    }

    components = []
    for comp_group in x_labels[model_name]:
        for comp in comp_group:
            components.append(comp)

    logit_formal_file = f"scores/logit_attribution/{model_name}/grouped_attr_formal_0_None.pt"
    formal_logits = t.load(logit_formal_file, map_location=t.device(device))
    mean_formal_logits = get_mean_sentence_tensor(formal_logits)
    df_formal_attr = pd.DataFrame(mean_formal_logits.numpy().T, columns=['DLA Idiom Formal', 'DLA Literal Formal'])
    
    std_formal_logits = t.std(formal_logits, dim = 0)
    df_formal_attr["DLA Std Idiom Formal"] = std_formal_logits[0]
    df_formal_attr["DLA Std Literal Formal"] = std_formal_logits[1]
    df_comp = pd.Series(components, name = "Component")

    logit_trans_file = f"scores/logit_attribution/{model_name}/grouped_attr_trans_0_None.pt"
    trans_logits = t.load(logit_trans_file, map_location=t.device(device))
    mean_trans_logits = get_mean_sentence_tensor(trans_logits)
    df_trans_attr = pd.DataFrame(mean_trans_logits.numpy().T, columns=['DLA Idiom Trans', 'DLA Literal Trans'])

    std_trans_logits = t.std(trans_logits, dim = 0)
    df_trans_attr["DLA Std Idiom Trans"] = std_trans_logits[0]
    df_trans_attr["DLA Std Literal Trans"] = std_trans_logits[1]

    idiom_score_formal_file = f"scores/idiom_scores/{model_name}/idiom_only_formal_0_None.pt"
    formal_idiom_comps = t.load(idiom_score_formal_file, map_location=t.device(device))
    formal_idiom_score = t.sigmoid(t.sum(formal_idiom_comps, dim = -1))
    mean_formal_idiom_score = get_lh_mean_scores(formal_idiom_score)
    df_formal_idiom_score = pd.DataFrame({"Component": list(mean_formal_idiom_score.keys()), "Idiom Score Formal": list(mean_formal_idiom_score.values())})
    std_formal_idiom_score = get_lh_std_scores(formal_idiom_score)
    df_std_formal_idiom_score = pd.DataFrame({"Component": list(std_formal_idiom_score.keys()), "Idiom Score Std Formal": list(std_formal_idiom_score.values())})
    # Merge std und mean: pd.merge(df1, df2.rename(columns={'id1':'id'}), on='id',  how='left')
    df_formal_idiom_score = pd.merge(df_formal_idiom_score, df_std_formal_idiom_score, on="Component",  how='left')

    idiom_score_trans_file = f"scores/idiom_scores/{model_name}/idiom_only_trans_0_None.pt"
    trans_idiom_comps = t.load(idiom_score_trans_file, map_location=t.device(device))
    trans_idiom_score = t.sigmoid(t.sum(trans_idiom_comps, dim = -1))
    mean_trans_idiom_score = get_lh_mean_scores(trans_idiom_score)
    df_trans_idiom_score = pd.DataFrame({"Component": list(mean_trans_idiom_score.keys()), "Idiom Score Trans": list(mean_trans_idiom_score.values())})
    #df_trans_idiom_score["Idiom Score Std Trans"] = get_lh_std_scores(trans_idiom_score)
    std_trans_idiom_score = get_lh_std_scores(trans_idiom_score)
    df_std_trans_idiom_score = pd.DataFrame({"Component": list(std_trans_idiom_score.keys()), "Idiom Score Std Trans": list(std_trans_idiom_score.values())})
    df_trans_idiom_score = pd.merge(df_trans_idiom_score, df_std_trans_idiom_score, on="Component",  how='left')

    df = pd.concat([df_comp, df_formal_idiom_score, df_trans_idiom_score, df_formal_attr, df_trans_attr], axis=1).set_index("Component")
    df["Idiom Score Diff"] = df["Idiom Score Formal"] - df["Idiom Score Trans"]

    df["DLA Diff Formal"] = df["DLA Idiom Formal"] - df["DLA Literal Formal"]
    df["DLA Diff Trans"] = df["DLA Idiom Trans"] - df["DLA Literal Trans"]
    df["DLA Diff Idiom"] = df["DLA Idiom Formal"] - df["DLA Idiom Trans"]
    df["DLA Diff Literal"] = df["DLA Literal Formal"] - df["DLA Literal Trans"]
    
    df.to_csv(f"plots/{model_name}/{model_name}.csv", index_label = "Index") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='idiom head detector')
    parser.add_argument('--model_name', '-m', help='model to run the experiment with', default="pythia-1.4b")
    parser.add_argument('--tensor_file', '-t', help='file with the tensor scores', default="./components/idiom_components/pythia-1.4b/idiom_only_formal_0_None.pt", type=str)
    parser.add_argument('--image_file', '-i', help='output file for the plot', default=None, type=str)
    parser.add_argument('--scatter_file', '-s', help='file with tensor scores for the scatter plot', default=None, type=str)

    model_name = parser.parse_args().model_name
    tensor_file = parser.parse_args().tensor_file
    img_file = parser.parse_args().image_file
    scatter_file = parser.parse_args().scatter_file
    device = "cuda" if t.cuda.is_available() else "cpu"

    os.makedirs(f"./plots/{model_name}", exist_ok=True)

    create_csv(model_name, device)

    # loaded_tensor = t.load(tensor_file, map_location=t.device(device))
    # #loaded_tensor = t.sigmoid(t.sum(loaded_tensor, dim = -1))
    # print(f"Loaded tensor with size: {loaded_tensor.size()}")
    # #plot_all(loaded_tensor, img_file, model_name, scatter_file)

    # if tensor_file.endswith("_comp.pt"):
    #     os.makedirs(f"./plots/{model_name}/components", exist_ok=True)
    #     plot_all_components(loaded_tensor, img_file, model_name)
    # elif "grouped" in tensor_file:
    #     path = f"./plots/{model_name}/logit"
    #     os.makedirs(path, exist_ok=True)
    #     x_labels = {
    #         "pythia-14m": ["L0H0", "L0H1", "L0H2", "L0H3", "L1H0", "L1H1", "L1H2", "L1H3", "L2H0", "L2H1", "L2H2", "L2H3", "L3H0", "L3H1", "L3H2", "L3H3", "L4H0", "L4H1", "L4H2", "L4H3", "L5H0", "L5H1", "L5H2", "L5H3", "0_mlp_out", "1_mlp_out", "2_mlp_out", "3_mlp_out", "4_mlp_out", "5_mlp_out", "embed", "bias"],
    #         "pythia-1.4b": [['L0H0', 'L0H1', 'L0H2', 'L0H3', 'L0H4', 'L0H5', 'L0H6', 'L0H7', 'L0H8', 'L0H9', 'L0H10', 'L0H11', 'L0H12', 'L0H13', 'L0H14', 'L0H15'], ['L1H0', 'L1H1', 'L1H2', 'L1H3', 'L1H4', 'L1H5', 'L1H6', 'L1H7', 'L1H8', 'L1H9', 'L1H10', 'L1H11', 'L1H12', 'L1H13', 'L1H14', 'L1H15'], ['L2H0', 'L2H1', 'L2H2', 'L2H3', 'L2H4', 'L2H5', 'L2H6', 'L2H7', 'L2H8', 'L2H9', 'L2H10', 'L2H11', 'L2H12', 'L2H13', 'L2H14', 'L2H15'], ['L3H0', 'L3H1', 'L3H2', 'L3H3', 'L3H4', 'L3H5', 'L3H6', 'L3H7', 'L3H8', 'L3H9', 'L3H10', 'L3H11', 'L3H12', 'L3H13', 'L3H14', 'L3H15'], ['L4H0', 'L4H1', 'L4H2', 'L4H3', 'L4H4', 'L4H5', 'L4H6', 'L4H7', 'L4H8', 'L4H9', 'L4H10', 'L4H11', 'L4H12', 'L4H13', 'L4H14', 'L4H15'], ['L5H0', 'L5H1', 'L5H2', 'L5H3', 'L5H4', 'L5H5', 'L5H6', 'L5H7', 'L5H8', 'L5H9', 'L5H10', 'L5H11', 'L5H12', 'L5H13', 'L5H14', 'L5H15'], ['L6H0', 'L6H1', 'L6H2', 'L6H3', 'L6H4', 'L6H5', 'L6H6', 'L6H7', 'L6H8', 'L6H9', 'L6H10', 'L6H11', 'L6H12', 'L6H13', 'L6H14', 'L6H15'], ['L7H0', 'L7H1', 'L7H2', 'L7H3', 'L7H4', 'L7H5', 'L7H6', 'L7H7', 'L7H8', 'L7H9', 'L7H10', 'L7H11', 'L7H12', 'L7H13', 'L7H14', 'L7H15'], ['L8H0', 'L8H1', 'L8H2', 'L8H3', 'L8H4', 'L8H5', 'L8H6', 'L8H7', 'L8H8', 'L8H9', 'L8H10', 'L8H11', 'L8H12', 'L8H13', 'L8H14', 'L8H15'], ['L9H0', 'L9H1', 'L9H2', 'L9H3', 'L9H4', 'L9H5', 'L9H6', 'L9H7', 'L9H8', 'L9H9', 'L9H10', 'L9H11', 'L9H12', 'L9H13', 'L9H14', 'L9H15'], ['L10H0', 'L10H1', 'L10H2', 'L10H3', 'L10H4', 'L10H5', 'L10H6', 'L10H7', 'L10H8', 'L10H9', 'L10H10', 'L10H11', 'L10H12', 'L10H13', 'L10H14', 'L10H15'], ['L11H0', 'L11H1', 'L11H2', 'L11H3', 'L11H4', 'L11H5', 'L11H6', 'L11H7', 'L11H8', 'L11H9', 'L11H10', 'L11H11', 'L11H12', 'L11H13', 'L11H14', 'L11H15'], ['L12H0', 'L12H1', 'L12H2', 'L12H3', 'L12H4', 'L12H5', 'L12H6', 'L12H7', 'L12H8', 'L12H9', 'L12H10', 'L12H11', 'L12H12', 'L12H13', 'L12H14', 'L12H15'], ['L13H0', 'L13H1', 'L13H2', 'L13H3', 'L13H4', 'L13H5', 'L13H6', 'L13H7', 'L13H8', 'L13H9', 'L13H10', 'L13H11', 'L13H12', 'L13H13', 'L13H14', 'L13H15'], ['L14H0', 'L14H1', 'L14H2', 'L14H3', 'L14H4', 'L14H5', 'L14H6', 'L14H7', 'L14H8', 'L14H9', 'L14H10', 'L14H11', 'L14H12', 'L14H13', 'L14H14', 'L14H15'], ['L15H0', 'L15H1', 'L15H2', 'L15H3', 'L15H4', 'L15H5', 'L15H6', 'L15H7', 'L15H8', 'L15H9', 'L15H10', 'L15H11', 'L15H12', 'L15H13', 'L15H14', 'L15H15'], ['L16H0', 'L16H1', 'L16H2', 'L16H3', 'L16H4', 'L16H5', 'L16H6', 'L16H7', 'L16H8', 'L16H9', 'L16H10', 'L16H11', 'L16H12', 'L16H13', 'L16H14', 'L16H15'], ['L17H0', 'L17H1', 'L17H2', 'L17H3', 'L17H4', 'L17H5', 'L17H6', 'L17H7', 'L17H8', 'L17H9', 'L17H10', 'L17H11', 'L17H12', 'L17H13', 'L17H14', 'L17H15'], ['L18H0', 'L18H1', 'L18H2', 'L18H3', 'L18H4', 'L18H5', 'L18H6', 'L18H7', 'L18H8', 'L18H9', 'L18H10', 'L18H11', 'L18H12', 'L18H13', 'L18H14', 'L18H15'], ['L19H0', 'L19H1', 'L19H2', 'L19H3', 'L19H4', 'L19H5', 'L19H6', 'L19H7', 'L19H8', 'L19H9', 'L19H10', 'L19H11', 'L19H12', 'L19H13', 'L19H14', 'L19H15'], ['L20H0', 'L20H1', 'L20H2', 'L20H3', 'L20H4', 'L20H5', 'L20H6', 'L20H7', 'L20H8', 'L20H9', 'L20H10', 'L20H11', 'L20H12', 'L20H13', 'L20H14', 'L20H15'], ['L21H0', 'L21H1', 'L21H2', 'L21H3', 'L21H4', 'L21H5', 'L21H6', 'L21H7', 'L21H8', 'L21H9', 'L21H10', 'L21H11', 'L21H12', 'L21H13', 'L21H14', 'L21H15'], ['L22H0', 'L22H1', 'L22H2', 'L22H3', 'L22H4', 'L22H5', 'L22H6', 'L22H7', 'L22H8', 'L22H9', 'L22H10', 'L22H11', 'L22H12', 'L22H13', 'L22H14', 'L22H15'], ['L23H0', 'L23H1', 'L23H2', 'L23H3', 'L23H4', 'L23H5', 'L23H6', 'L23H7', 'L23H8', 'L23H9', 'L23H10', 'L23H11', 'L23H12', 'L23H13', 'L23H14', 'L23H15'], ['0_mlp_out', '1_mlp_out', '2_mlp_out', '3_mlp_out', '4_mlp_out', '5_mlp_out', '6_mlp_out', '7_mlp_out', '8_mlp_out', '9_mlp_out', '10_mlp_out', '11_mlp_out', '12_mlp_out', '13_mlp_out', '14_mlp_out', '15_mlp_out', '16_mlp_out', '17_mlp_out', '18_mlp_out', '19_mlp_out', '20_mlp_out', '21_mlp_out', '22_mlp_out', '23_mlp_out'], ['embed', 'bias']]
    #     }
    #     mean_logit_attr = get_mean_sentence_tensor(loaded_tensor)
    #     seen_comps = 0
    #     for comp_group in range(len(x_labels[model_name])):
    #         num_comps = len(x_labels[model_name][comp_group])
    #         comp_logit = mean_logit_attr[:, seen_comps:(seen_comps+num_comps)]
    #         plot_logit_attribution_split(comp_logit, x_labels = x_labels[model_name][comp_group], filename=f"{path}/{img_file}_{comp_group}.png")
    #         seen_comps += len(x_labels[model_name][comp_group])
    # else:
    #     os.makedirs(f"./plots/{model_name}/score", exist_ok=True)
    #     plot_all(loaded_tensor, img_file, model_name, scatter_file)


    
