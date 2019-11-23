```sh
docker build . -t tf-ga-reader
```

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

```sh
python main.py --data_dir /data/bioread --embed_file /data/w2v/pubmed2018_w2v_200D.txt
```
