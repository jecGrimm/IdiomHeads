# IdiomHeads
This repository is part of the Master thesis "Spilling the beans: Interpreting Attention Patterns for Idioms in transformer-based Language Models".

## Todos
1. Head detection
- Formal scores
    - fix idiom pos ngram 50 formal 
    - every score speichern und erst dann avg nehmen
    - schneller machen
        - Huggingface
- translated scores
- static scores
- literal scores
- plots: 
    - line
    - boxplot
    - heatmap
    - histogram
2. Logit attribution
3. Ablation
4. Schreiben
- Deutsches Abstract
- Related Work
- Experiment
- Evaluation
- Discussion
- Conclusion
5. Code cleanup
- Readme
- Notebook übertragen
- Environment

### Optional
- Scatter Plot feature vectors
- Regression: Add weights and bias
- Clustering
- Loss für static Idiomsätze
- Token for whole idiom in vocabulary


## Usage
### Environment
conda activate idiom -> python 3.12.7, geht nicht mit pytorch 1.11.0 (wie in transformer lens doku angegeben), stattdessen mit pytorch generell installiert
