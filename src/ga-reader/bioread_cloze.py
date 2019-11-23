from utils import multi_replace, split_instance
from nltk import word_tokenize
from tqdm import tqdm

from itertools import islice
from pathlib import Path


BIOREAD_LITE_DIR = Path.home()/'bioread_lite'
VALID_DIR = BIOREAD_LITE_DIR/'valid'

FAKE_URL = 'http://some_url'


def convert_to_cnn_daily_mail(text):
    context, question, entities, answer = split_instance(text)
    return '\n\n'.join((
        FAKE_URL,
        parse_context(context),
        parse_question(question),
        parse_answer(answer),
        parse_entities(entities)
    ))


def tokenize(text):
    tokenized = word_tokenize(multi_replace(text, [
        ('@entity', 'atentity'),
        ('@placeholder', 'atplaceholder')
    ]))
    return multi_replace(' '.join(tokenized), [
        ('atentity', '@entity'),
        ('atplaceholder', '@placeholder')
    ])


def parse_context(text):
    return tokenize(text)


def parse_question(text):
    return tokenize(text.replace('XXXXXX', '@placeholder'))


def parse_entities(text):
    lines = text.strip().split('\n')
    lines = (line.strip() for line in lines)
    return '\n'.join(lines)


def parse_answer(text):
    return text.strip().split(':')[0].strip()


def convert_folder(src, dst, n=None):
    for file in tqdm(list(islice(src.glob('xx*'), n))):
        content = convert_to_cnn_daily_mail(file.read_text())
        new_file = dst/f'{file.stem}.question'
        new_file.write_text(content)


def main():
    convert_folder(
        BIOREAD_LITE_DIR/'test',
        BIOREAD_LITE_DIR/'questions'/'test',
    )


if __name__ == '__main__':
    main()
