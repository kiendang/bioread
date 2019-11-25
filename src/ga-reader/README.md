# Code for training Gated Attention Reader on Bioread dataset

## Prepare dataset and word embeddings

The Bioread and BioreadLite datasets can be obtained [here](https://archive.org/details/bioread_dataset.tar) (from http://nlp.cs.aueb.gr/publications.html), word embeddings (trained on MEDLINE®/PubMed®) [here](https://archive.org/details/pubmed2018_w2v_200D.tar) (from http://nlp.cs.aueb.gr/software.html)

Data need to be converted to CNN/Daily Mail format using `scripts/bioread_cloze.py`.

Embeddings need to be converted from binary to text format using `scripts/w2v2glove.py`.

## Run the code

Requirements: `docker`, `nvidia-container-toolkit` or `nvidia-docker2`.

### Build the image

```sh
docker build . -t tf-ga-reader
```

### Run the container

```sh
docker run --gpus all \
    -u "$UID" \
    -v /etc/passwd:/etc/passwd \
    -v "$W2V_DIR":/data/w2v \
    -v "$BIOREAD_DIR":/data/bioread \
    -v $(pwd)/logs:/logs \
    -v $(pwd)/model:/model \
    --rm -it \
    tf-ga-reader /bin/bash
```

where `W2V_DIR` and `BIOREAD_DIR` are paths to the directories containing the word embeddings and dataset respectively.

If using `docker<=19.03` and/or using `nvidia-docker2` instead of `nvidia-container-toolkit`, swap `--runtime=nvidia` for `--gpus`

### Train the model

```sh
python main.py --data_dir /data/bioread --embed_file /data/w2v/pubmed2018_w2v_200D.txt
```
