from preprocess import preprocess

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from itertools import islice
from pathlib import Path


DATA_DIR = Path('/home/kien/bioread_lite/sample/')
DATA_DIR_LITE = Path('/home/kien/bioread_lite/lite/')

directories_full = {
    'valid': DATA_DIR/'validation',
    'train': DATA_DIR/'training',
    'test': DATA_DIR/'test'
}

directories_lite = {
    'valid_lite': DATA_DIR_LITE/'validation',
    'train_lite': DATA_DIR_LITE/'training',
    'test_lite': DATA_DIR_LITE/'test'
}

directories = {**directories_lite, **directories_full}

for path in directories.values():
    assert path.exists()

BERT = Path('/home/kien/biobert_v1.0_pubmed_pmc/')

CACHE = Path('cache')


def main():
    tokenizer = BertTokenizer.from_pretrained(BERT)

    for name, path in directories.items():
        print(f"Preprocessing {path}...")
        torch.save(
            [
                preprocess(f.read_text(), tokenizer, max_seq_length=512)
                for f in tqdm(list(path.glob('xx*')))
            ],
            CACHE/f"{name}.pt"
        )


if __name__ == '__main__':
    main()
