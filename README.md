# IdiomHeads
This repository is part of the Master thesis "Spilling the beans: Interpreting Attention Patterns for Idioms in transformer-based Language Models".

## Todos
0. Awareness
- static pythia 2570 DONE
1. Head detection
- idiom score
    - static pythia 2917 failed because of long sentence 1773 -> 3500 (alle Sätze wegen new idiom pos) failed ohne Error bei 654 -> 3597 failed weil idiom positions nicht mit Datenlänge übereinstimmen (ab 654) -> 3599 (fixed idiom pos ab 654) failed weil der Hf-Data sich wieder in ein Dict umgewandelt hat -> 3600 (full run) failed Begrenzung für idiom pos vergessen -> 3601 (fix idiom pos begrenzung) failed OOM -> 3743 (ab 1817) failed OOM -> 3751 DONE
    - plot: mean dots, Farben nicht nach Heads, sondern wie hoch die Logit Attribution ist
- literal score
    - static 3809 DONE
2. Logit attribution
    - static 4205 DONE
    - Warum sind Bias und Embeds bei Pythia und Llama so konträr?
3. Ablation
    - Zusammenspiel von Heads abschalten, die für Idiome sind: 2642 failed segmentation error -> 2645 failed trans switch DONE
    - trans 2885 (fixed trans switch) failed None during clean -> 2961 (fixed single token clean run) DONE
    - static 4252 (ohne trans) failed weil es kein dir mit ablation_name gibt -> 4258 (fix makedir ablation)
4. zweites Modell: qwen oder olmo oder llama
    - nwp
        - llama 2565 failed cli None -> 2566 (fix none error) failed cli.start still in compute_awareness -> 2567 failed start und end verwechselt -> 2568 DONE
    - Head detection
        - idiom score 2569 DONE 
        - literal score 2886 failed short model name -> 2890 (full model name), hat ohne Fehlermeldung nach formal 2282 aufgehört -> 3475 (formal ab 2285, trans full) -> 3513 (nur trans) DONE
    - logit attribution - 3742 (formal und trans) failed OOM 1280 -> 3755 (ab 1280) OOM bei 1280 -> 3771 (nur formal, ab 1281 mit alter pos file) OOM für 1281 -> 3772 (ab 1282) OOM für 1282 -> 3773 (ab 1283) OOM für 1386 -> 3775 (ab 1387) OOM für 2210 -> 3377 (2211) DONE
        - trans: 3778 (nur trans, pop lange sents neue idiom pos) OOM für 1383 -> 3779 (ab 1384) OOM für 2205 -> 3780 (ab 2206) OOM für 2206 -> 3781 (ab 2207) DONE
    - llm transparency - DONE?
    - ablation - 3789 (full)
        - mit L15H12 für idiom diff nochmal laufen lassen -> 9615
        - plot: Heatmap für die Beispielsätze
5. Drittes Modell: Tiny
    - nwp
        - formal und trans 3602 DONE (commited auf remote main)
        - static
    - Head detection
        - idiom score 
            - formal 3744 DONE
            - trans 3744 failed weil argmax idiom_fractions leer -> 3811 failed weil idiom_pos zu lang -> 4210
        - literal score
    - logit attribution
    - llm transparency ?
    - ablation
5. Schreiben
- Selbstständigkeitserklärung -> auch digital unterschreiben? Muss das die neue sein?
- Related Work
    - muss noch Logit Lens rein
    - hier müssen noch TransformerLens und LM Transparency Tool rein
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
- nwg und loss Experimente in Repo übertragen
- Angeben, dass h11-Update auf 0.16.0 von dependabot gemacht wurde
- fix plots
- fix distribution 
- alle Tensoren sollten die gleiche Größe haben
- alle Funktionen testen (besonders plots)
- Sigmoid in allen Skripten gleich

### Optional
- Duplizierende Methoden (eigl bereits abgedeckt):
    - Clustering
    - weitere Modelle
        - Experimente mit pythia-14m vergleichen
    - weitere Daten
        - static scores
        - ambigue scores
        - random scores
        - transparent vs. opaque (Madabushi 2022)

- Noch nicht abgedeckte Methoden:
    - Aktivierungen austauschen und schauen, was das mit der Prediction macht
    - Letztes Idiomwort austauschen mit trans und random und Unterschied betrachten
        - Loss-Unterschiede
        - logit Difference
        - cosine similarity 
    - Cosine Similarity zwischen den Pattern berechnen
        - Paare finden, die gleich lang sind (formal und trans)
        - ganze Pattern
        - letztes Query word
        - letztes query idiom word
        - random Satz, der gleich lang ist
        - Gruppen von Idiomen vergleichen
            - dimension reduction cluster: tsn-e, enge oder weite Cluster?
            - pairwise cosine 
                - jedes Idiom mit jedem und dann den Durchschnitt für jeden Head
    - Route im LLM-transparency-tool anschauen
    - logit lens oder utils.test_prompt 

- Änderungen an bestehenden Methoden:
    - trans scores: momentan nur für idiom_pos, sollte aber eigl die gesamte Idiomübersetzung verwenden?
    - model mit from_pretrained_no_processing laden und überprüfen, ob das die Scores ändert
    - literal score relativieren mit idiom score?
    - idiom score mit generellem ngram-Score relativieren oder den einzelnen Features?

- Code clean:
    - idiom positions in data class verschieben
    - line single plot: alle layer anzeigen
    - hist plot: alle Zahlen anzeigen
    - alle plots in plot all
    - cache: to token rausnehmen
    - score überklasse
    - Wichtige Plots in main Funktion aufrufen
    - literal und idiom score methoden gleich benennen
    - in idiom scores ordner nur die sigmoid-Scores speichern


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
pip install IPython

Wenn Fehlermeldung kommt, dass torchvision:nms nicht vorhanden ist: 
conda remove torchvision
`conda install pytorch torchdata torchvision -c pytorch -y`

Mit conda yml file: 
conda env create -f environment.yml (kommt ein pip Fehler, dass torchaudio nicht geht)
conda activate idiom
`conda install pytorch torchdata torchvision -c pytorch -y` (kommt Nachricht, dass alles schon installiert ist, funzt aber trotzdem)
pip install transformer-lens
pip install merge-tokenizers
pip install IPython

für LLMTransparency:
cd llm-transparency-tool
conda env update --name idiom --file env.yaml
pip install -e .
pip install streamlit
pip install streamlit_extras
pip install pyinstrument

huggingface-cli login <huggingface_token> -> Access zu Llama 3 nötig

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
sbatch --partition=NvidiaAll ./scripts/idiom_only_trans.sh
sbatch --partition=NvidiaAll ./scripts/literal_only.sh
sbatch --partition=NvidiaAll ./scripts/idiom_only.sh
sbatch --partition=NvidiaAll ./scripts/logit_attr.sh
sbatch --partition=NvidiaAll ./scripts/awareness.sh
sbatch --partition=NvidiaAll ./scripts/cage_pythia.sh
sbatch --partition=NvidiaAll ./scripts/cage_qwen.sh
sbatch --partition=NvidiaAll ./scripts/ablation.sh
sbatch --partition=NvidiaAll ./scripts/contribution.sh
sbatch --partition=NvidiaAll ./scripts/idiom_only_tiny.sh

NICHT VERGESSEN, CONDA ZU AKTIVIEREN!

See jobs: sacct
Cancel jobs: scancel <jobnumber>

## Zeiten
pythia 14m 
- formal 
    - idiom 
        - b 3: 42,78s/ex
        - b 1: 2.86 - 43,53s/ex (delete tensors), 3.89 - 132.81 (load model with float16), 2.80 s/ex(caging), 2.35s/ex (no caging)
    - mean
        - b3: 4,55s/ex
        - b1, GPU: 12s/ex, 1.2s/ex (extract tensor per attention pattern)
        - b1: 4.04s/ex, 2.72s/ex (del tensors, load model as float 16)
    - idiom only
        - b1: 1.08s/ex, 1,57s/ex (load model with float16)
    - literal only
        - b1: 2.88a/ex (load model with bfloat16, delete cache)
        - b1, GPU: 1.3s/ex
    - logit 
        - b1: 5.17s/ex, 9.57 s/ex (copied logit_attrs, bfloat16)
- trans 
    - idiom
        - b 8: 1721 s/ex
        - only comp mit contr
            - b1: 2.71 - 22.32 s/ex
    - literal only
        - b1: 2.2s/ex
    - logit 
        - b1: 8.15s/ex

pythia 1.4b - immer mit GPU
- formal 
    - literal
        - b 3: cuda oom 
    - mean
        - b3: cuda oom
        - b1: cudo oom
        - b1: 5s/ex-41s/ex (extract tensor per attention pattern)
    - idiom only
        - b1: 18.28s/ex (load model with float16)
- trans literal
    - idiom
        - b 8: cuda oom
    - idiom only
        - b1: 19.39s/ex (load model with float16)

