# Code for adapting BERT to Bioread dataset

## Prepare

Download [BioBert](https://arxiv.org/abs/1901.08746) pretrained on PubMed abstracts and PMC full texts [here](https://drive.google.com/file/d/1jGUu2dWB1RaeXmezeJmdiPKQp3ZCmNb7/view?usp=sharing) (from https://github.com/naver/biobert-pretrained). The model is in `tensorflow` format originally. Convert to `pytorch` checkpoint following instructions [here](https://huggingface.co/transformers/converting_tensorflow_models.html).

## Preprocess data

Run `run_preprocess.py` to preprocess and cache data. Edit paths to data, pretrained model and cache accordingly. Input data are data converted to CNN/Daily Mail format using `scripts/bioread_cloze.py` script.

## Train model

Edit paths to preprocessed data, pretrained model and output model in `run_cloze_qa.py`. Hyperparameters can be adjusted as well. The ones in the file are what I used when training BioreadLite on 1 single V100 GPU for 2 epochs using 40000 examples in the training set (5%) for training and 5000 in the dev set (10%) for validation.

```sh
python run_cloze_qa.py --train
```

Training loss and validation accuracy can be monitored using `tensorboard`

```sh
tensorboard --logdir=runs
```

To score the model on test set run

```sh
python run_cloze_qa.py --test --model "$MODEL"
```

where `MODEL` is the path to the directory containing the trained model.
