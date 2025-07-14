# IdiomHeads
This repository is part of the Master Thesis "Spilling the beans: Interpreting Attention Patterns for Idioms in transformer-based Language Models". It contains experiments that measure the importance of attention heads in Pythia-1.4B and Llama-3.2-1B-Instruct when processing the idiomatic sentences in the Labelled EPIE dataset (Saxena et al. 2021).  

## Usage
### Installation
#### Clone
Clone and add submodules:<br> 
`git clone --recurse-submodules git@github.com:jecGrimm/IdiomHeads.git`

If the repository has already been cloned without initializing the submodules, please run <br>
`git submodule update --init --recursive` <br>
to add the submodules afterwards. Without this command, the directories `TransformerLens-intro` and `EPIE_Corpus` are empty.

#### Environment
Install the required packages with conda:<br>
`conda env create -f environment.yml`<br>

Activate environement:<br>
`conda activate idiom`<br>

For access to Llama-3.2-1B-Instruct:<br>
`huggingface-cli login <huggingface_token>`

For the LM Transparency Tool:
```bash
cd llm-transparency-tool
conda env update --name idiom --file env.yaml
pip install -e .
pip install streamlit
pip install streamlit_extras
pip install pyinstrument
```

#### Execution
All experiments can be run with the command line interface arguments specified in `cli.py`. The plots can be created with the CLI in the main function of the script `plots.py`. <br>

For the LM Transparency Tool, first copy the file `local.json` from the folder `resources/Llama-3.2-1B-Instruct` to the folder `llm-transparency-tool/config`. Copy the file `sents.txt` from the folder `resources/Llama-3.2-1B-Instruct` to the folder `llm-transparency-tool`. Execute the LM Transparency Tool with the following commands:<br>
```bash
cd llm-transparency-tool
streamlit run llm_transparency_tool/server/app.py -- config/local.json
``` 
Please make sure that you have access to Llama-3.2-1B-Instruct via Hugging Face and that the required access token is active before executing the tool. 

## Content
### EPIE_Corpus
This directory contains the Labelled EPIE dataset (Saxena et al. 2021)

### future_work
This directory contains future projects that are not completed yet. 

### gpu_scripts
This directory contains the SLURM-scripts and the output of the experiment runs.

### llm-transparency-tool
This directory contains the code for the LM Transparency Tool (Tufanov et al. 2022) that is used for the qualitative analysis.

### merge-tokenizers
This directory contains the code for the library merge-tokenizers (https://github.com/symanto-research/merge-tokenizers) that is used to align the sentences tokenized by EPIE and the model.

### plots
This directory contains all plots organized by the examined models.

#### ablation
This directory contains the plots for the ablation study.

#### idiom_components
This directory contains the plots for the features of the Idiom Score.

#### idiom_scores
This directory contains the plots for the Idiom Score.

#### literal_components
This directory contains the plots for the features of the Literal Score.

#### literal_scores
This directory contains the plots for the Literal Score.

#### logit_attribution
This directory contains the plots for the DLA scores.

#### loss
This directory contains the plots for the loss scores.

#### results_{model_id}.csv
This file contains the scores for each head and experiment.

### resources
This directory contains additional resources that can be used for the experiments organized by model. 

#### formal
This directory contains the original idiom predictions for the next word generation task.

#### trans
This directory contains the original translated predictions for the next word generation task.

#### qual_analysis
This directory contains the sentences and the configuration for the qualitative analysis conducted with the LM Transparency Tool. This tool cannot be executed for models of the Pythia suite.

### scores
This directory contains the results of each experiment. The scores are stored in tensor files. Each tensor entails one score per head and sentence.

#### ablation
This directory contains the results of the ablation study.

#### awareness
This directory contains the results of the awareness experiment.

#### idiom_components
This directory contains the calculated features of the Idiom Score.

#### idiom_scores
This directory contains the Idiom Scores.

#### literal_components
This directory contains the calculated features of the Literal Scores.

#### literal_scores
This directory contains the Literal Scores.

#### logit_attribution
This directory contains the DLA scores.

#### loss
This directory contains the loss scores.

### TransformerLens-intro
This directory contains the code for the library TransformerLens (Nanda et al. 2022). The experiments are based on this library.

### visualizations
This directory contains additional visualizations. The attention pattern visualizations can be created with the notebook `visualize_pattern.ipynb`. The contribution pattern can be created with the LM Transparency Tool.

### ablation.py
This script contains the class Ablation for the ablation study.

### cage.py
This script contains the class Cage. This class can be used to cage attention parameters.

### cli.py
This script contains the class CLI for the experiment cli.

### compute_ablation.py
This script executes the ablation study.

### compute_contribution.py
This script calculates the contribution scores as implemented in the LM Transparency Tool. These are part of a side-experiment but require too many computational resources and are therefore excluded from the Thesis. 

### compute_idiom_awareness.py
This script executes the awareness experiment.

### compute_idiom_score.py
This script calculates the Idiom Scores.

### compute_literal_score.py
This script calculates the Literal Scores.

### compute_logit_attr.py
This script calculates the DLA scores.

### contribution.py
This script defines the class Contribution which calculates the contribution scores in the same manner as the LM Transparency Tool.

### data.py
This script defines the class EPIE_Data which loads the Labelled EPIE dataset and preprocesses it. The data is transformed to Hugging Face Datasets. Computationally expensive and literal sentences are excluded from the experiments. 

### environment.yml
This file contains the setup of the environment `idiom` which is required for the excecution of the scripts in this repository.

### helper.py
This script contains functions that support the analysis of the experiment results.

### idiom_awareness.py
This script defines the class IdiomAwareness which is used for the awareness experiment.

### idiom_score.py
This script defines the class IdiomScorer which is used for the calculation of the Idiom Scores.

### literal_score.py
This script defines the class LiteralScorer which is used for the calculation of the Literal Scores.

### logit_attribution.py
This script defines the class LogitAttribution which is used for the calculation of the DLA scores.

### plot.py
This script contains functions that plot the results of the experiments.

## Bibliography
Neel Nanda and Joseph Bloom. “TransformerLens.”, 2022. https://github.com/TransformerLensOrg/TransformerLens.

Saxena, Prateek, and Soma Paul. “Labelled EPIE: A Dataset for Idiom Sense Disambiguation.” In Text, Speech, and Dialogue, edited by Kamil Ekštein, František Pártl, and Miloslav Konopík, 210–21. Cham: Springer International Publishing, 2021. https://doi.org/10.1007/978-3-030-83527-9_18.

Tufanov, Igor, Karen Hambardzumyan, Javier Ferrando, and Elena Voita. “LM Transparency Tool: Interactive Tool for Analyzing Transformer Language Models.” In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), edited by Yixin Cao, Yang Feng, and Deyi Xiong, 51–60. Bangkok, Thailand: Association for Computational Linguistics, 2024. https://doi.org/10.18653/v1/2024.acl-demos.6.

AI-written code is marked directly in the documentation of the functions.


