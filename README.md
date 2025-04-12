# IdiomHeads
This repository is part of the Master thesis "Spilling the beans: Interpreting Attention Patterns for Idioms in transformer-based Language Models".

## Todos
1. Head detection
- Formal scores
   - full lrz
        - push remote scores
- translated scores
    - full lrz: 
        - push remote scores
- plots: 
    - line
        - 1.4b
            - formal idiom
            - trans idiom
            - formal literal
            - trans literal
    - boxplot
        - 1.4b
            - formal idiom
            - trans idiom
            - formal literal
            - trans literal
    - heatmap
        - 1.4b
            - formal idiom
            - trans idiom
            - formal literal
            - trans literal
    - histogram
        - 1.4b
            - formal idiom
            - trans idiom
            - formal literal
            - trans literal
- literal score
    - formal score
        - full
    - trans score
        - test 
        - full
    - plot scatter to see correlation
        - test
        - full
2. Logit attribution
3. Ablation
    - loss
    - nwg
4. zweites Modell
5. Schreiben
- Selbstständigkeitserklärung -> auch digital unterschreiben?
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
- Plot title
- unnötige Funktionen ins Archiv
- Wichtige Plots in main Funktion aufrufen
- literal score relativieren mit idiom score?
- idiom score mit generellem ngram-Score relativieren oder den einzelnen Features?

### Optional
- Scatter Plot feature vectors
- Regression: Add weights and bias
- Clustering
- Loss für static Idiomsätze
- Token for whole idiom in vocabulary
- static scores
- literal scores
- trans scores: momentan nur für idiom_pos, sollte aber eigl die gesamte Idiomübersetzung verwenden
- idiom positions in data class verschieben


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
sbatch --partition=NvidiaAll ./scripts/detect_formal.sh
sbatch --partition=NvidiaAll ./scripts/detect_small.sh
sbatch --partition=NvidiaAll ./scripts/compute_trans.sh
sbatch --partition=NvidiaAll ./scripts/literal_formal.sh
sbatch --partition=NvidiaAll ./scripts/literal_trans.sh
sbatch --partition=NvidiaAll ./scripts/mean_idiom_formal.sh

NICHT VERGESSEN, CONDA ZU AKTIVIEREN!

See jobs: sacct
Cancel jobs: scancel <jobnumber>

## Zeiten
pythia 14m 
- formal 
    - idiom 
        - b 3: 42,78s/ex
    - mean
        - b3: 4,55s/ex
- trans idiom
    - b 8: 1721 s/ex

pythia 1.4b
- formal literal
    - b 3: cuda oom 
- trans literal
    - b 8: cuda oom
