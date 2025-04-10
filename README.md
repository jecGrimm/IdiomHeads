# IdiomHeads
This repository is part of the Master thesis "Spilling the beans: Interpreting Attention Patterns for Idioms in transformer-based Language Models".

## Todos
1. Head detection
- Formal scores
   - full lrz
- translated scores
- plots: 
    - line
    - boxplot
    - heatmap
    - histogram
    - average
    - std
2. Logit attribution
3. Ablation
4. zweites Modell
5. Schreiben
- Neues Template
- Selbstständigkeitserklärung
- Deutsches Abstract
- Related Work
- Experiment
- Evaluation
- Discussion
- Conclusion
6. Code cleanup
- Readme
- Notebook übertragen
- Environment
- submodule merge-tokenizers entfernen
- move explore functions from scorer to plot 
- Update requirements and yml-file

### Optional
- Scatter Plot feature vectors
- Regression: Add weights and bias
- Clustering
- Loss für static Idiomsätze
- Token for whole idiom in vocabulary
- static scores
- literal scores


## Usage
### Installation
Clone and add submodules:<br> 
`git clone --recurse-submodules git@github.com:jecGrimm/GartenlaubeExtractor.git`

If the repository has already been cloned without initializing the submodules, please run <br>
`git submodule update --init --recursive` <br>
to add the submodules afterwards. Without this command, the directory `TransformerLens-intro` and `EPIE_Corpus` are empty.

### Installation
conda activate idiom -> python 3.12.7, geht nicht mit pytorch 1.11.0 (wie in transformer lens doku angegeben), stattdessen mit pytorch generell installiert

Schritte neu aufsetzen ohne conda environment file:
conda create -n idiom python==3.12.7
conda activate idiom
`conda install pytorch torchdata torchvision -c pytorch -y`
pip install -r conda_requirements.txt
pip install merge-tokenizers

Wenn Fehlermeldung kommt, dass torchvision:nms nicht vorhanden ist: 
conda remove torchvision
`conda install pytorch torchdata torchvision -c pytorch -y`

Mit conda yml file: 
conda env create -f environment.yml (kommt ein pip Fehler, dass torchaudio nicht geht)
conda activate idiom
`conda install pytorch torchdata torchvision -c pytorch -y` (kommt Nachricht, dass alles schon installiert ist, funzt aber trotzdem)
pip install transformer-lens
pip install merge-tokenizers

### SLURM
ssh grimmj@remote.cip.ifi.lmu.de
sbatch --partition=NvidiaAll script.sh


