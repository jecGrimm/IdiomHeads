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
            scores[f"{layer}.{head}"] = t.mean(sent_last[layer][head]).numpy()
    return scores

def get_lh_std_scores(tensor):
    scores = dict()
    sent_last = t.einsum("ijk->jki", tensor)
    for layer in range(tensor.size(1)):
        for head in range(tensor.size(2)):
            scores[f"{layer}.{head}"] = t.std(sent_last[layer][head]).numpy()
    return scores

def explore_scores(tensor, filename = None, model_name = None):
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

    mean_above_59 = {head: float(score) for head, score in mean_scores.items() if score >= 0.59}
    output += f"\nThere are {len(mean_above_59)} scores with a score higher than 0.59: {sorted(mean_above_59, key = lambda k:mean_above_59.get(k), reverse = True)}"
    # STD
    std_scores = get_lh_std_scores(tensor)
    std_above_59 = [(head, float(std_scores[head])) for head in mean_above_59.keys()]
    output += f"\nThese scores have a std of: {std_above_59}"
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

    if filename and model_name:
        with open(f"./plots/{model_name}/{filename}.txt", 'w', encoding = "utf-8") as f:
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

def plot_logit_attribution_split(logit_attr: t.Tensor, title: str = "", x_labels = None):
    y_labels = ["Idiom", "Literal"]
    return px.imshow(logit_attr.float(), title = title, x=x_labels, y=y_labels, labels={"x":"Component", "y":"Position", "color":"Mean logit attribution"}, color_continuous_midpoint=0.0)

def plot_all(tensor, filename = None, model_name = None, scatter_file = None):
    path = f"./plots/{model_name}"
    if filename and model_name:
        plot_tensor_line(tensor, f"{path}/mean_line_{filename}.png")
        plot_line_std(tensor, f"{path}/std_line_{filename}.png")
        plot_heatmap(tensor, f"{path}/heat_{filename}.png")
        plot_tensor_hist(tensor, f"{path}/hist_{filename}.png")
        explore_scores(tensor, filename, model_name)
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
            plot_scatter(loaded_tensor, scatter_tensor, f"plots/{model_name}/scatter_{filename}_literal.png")
        else:
            plot_scatter(loaded_tensor, scatter_tensor)
    
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
            explore_scores(tensor, filename, model_name)

        else:
            plot_tensor_line(tensor, title = f"Average {comp} (component) per head and layer", ylabel=comp)
            plot_line_std(tensor, title = f"Standard deviation of the {comp} (component) per head and layer", ylabel=comp)
            plot_heatmap(tensor, title = f"{comp[0].upper()}{comp[1:]} (component)")
            plot_tensor_hist(tensor, title = f"Distribution of the {comp} (component)", xlabel=comp)
            explore_scores(tensor)

    if orig_filename and model_name:
        plot_scatter_components(comp_dict, f"{path}/scatter_{orig_filename}_comp.png")
    else:
        plot_scatter_components(comp_dict)

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

    if not os.path.isdir("./plots"):
        os.mkdir("./plots")

    if not os.path.isdir(f"./plots/{model_name}"):
        os.mkdir(f"./plots/{model_name}")

    loaded_tensor = t.load(tensor_file, map_location=t.device(device))
    print(f"Loaded tensor with size: {loaded_tensor.size()}")

    if tensor_file.endswith("_comp.pt"):
        plot_all_components(loaded_tensor, img_file, model_name)
    else:
        plot_all(loaded_tensor, img_file, model_name, scatter_file)

    # get_head_info("1.1", loaded_tensor)
    # get_head_info("1.4", loaded_tensor)
    # get_head_info("1.6", loaded_tensor)
    # get_head_info("3.2", loaded_tensor)


    
