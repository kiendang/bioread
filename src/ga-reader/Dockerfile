FROM tensorflow/tensorflow:1.0.1-gpu-py3

RUN pip install --no-cache-dir tqdm

COPY gated-attention-reader /ga-reader
WORKDIR /ga-reader

RUN mkdir /data
