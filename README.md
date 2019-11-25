# Code for training NLP models on [Bioread](https://www.aclweb.org/anthology/L18-1439/) dataset

Available models are [Gated Attention Reader](https://arxiv.org/abs/1606.01549) and [BERT](https://arxiv.org/abs/1810.04805). Code for them is in `src/ga-reader` and `src/bert-lm` respectively.

## Prepare dataset

The Bioread and BioreadLite datasets can be obtained [here](https://archive.org/details/bioread_dataset.tar) (from http://nlp.cs.aueb.gr/publications.html).

Data need to be converted to [CNN/Daily Mail](https://arxiv.org/abs/1506.03340) format using `scripts/bioread_cloze.py`.