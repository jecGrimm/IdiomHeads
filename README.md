# IdiomHeads
This repository is part of the Master thesis "Spilling the beans: Interpreting Attention Patterns for Idioms in transformer-based Language Models".

## Todos
1. Head detection
- literal score
    - formal score
        - full
    - trans score
        - test 
        - full
    - plot scatter to see correlation
        - test
        - full
- plots: 
    - line
        - 1.4b
            - formal literal
            - trans literal
    - heatmap
        - 1.4b
            - formal literal
            - trans literal
    - histogram
        - 1.4b
            - formal literal
            - trans literal
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
- literal und idiom score methoden gleich benennen
- score überklasse

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
- line single plot: alle layer anzeigen
- hist plot: alle Zahlen anzeigen


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
sbatch --partition=NvidiaAll ./scripts/mean_idiom_trans.sh
sbatch --partition=NvidiaAll ./scripts/idiom_only_formal.sh

NICHT VERGESSEN, CONDA ZU AKTIVIEREN!

See jobs: sacct
Cancel jobs: scancel <jobnumber>

## Zeiten
pythia 14m 
- formal 
    - idiom 
        - b 3: 42,78s/ex
        - b 1: 2.86 - 43,53s/ex (delete tensors), 3.89 - 132.81 (load model with float16)
    - mean
        - b3: 4,55s/ex
        - b1, GPU: 12s/ex, 1.2s/ex (extract tensor per attention pattern)
        - b1: 4.04s/ex, 2.72s/ex (del tensors, load model as float 16)
    - idiom only
        - b1: 1.08s/ex, 1,57s/ex (load model with float16)
    - literal only
        - b1: 2.88a/ex (load model with bfloat16, delete cache)
- trans idiom
    - b 8: 1721 s/ex

pythia 1.4b
- formal 
    - literal
        - b 3: cuda oom 
    - mean
        - b3: cuda oom
        - b1: cudo oom
        - b1, GPU: 5s/ex-41s/ex (extract tensor per attention pattern)
    - idiom only
        - b1: 1.21s/ex (load model with float16)
- trans literal
    - b 8: cuda oom

