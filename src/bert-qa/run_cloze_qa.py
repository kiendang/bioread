from utils import Example

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange

from transformers import AdamW, get_linear_schedule_with_warmup
from modeling_cloze_qa import BertForClozeQA

import argparse
import logging
from operator import getitem
from pathlib import Path
from typing import List


BERT = Path('/home/kien/biobert_v1.0_pubmed_pmc/')
CACHE = Path('cache')
MODEL = Path('model')


batch_size = 2
num_train_epochs_set = 2
learning_rate = 5e-5
adam_epsilon = 1e-8
warmup_steps = 0
max_steps = -1
gradient_accumulation_steps = 1
weight_decay = 0.0
max_grad_norm = 1.0
eval_steps = 1000
logging_steps = 100
device = torch.device('cuda:0')


def pad(tensor, x, y):
    xs, ys = tensor.size()
    with torch.no_grad():
        paddings = (0, y - ys, 0, x - xs)
        result = F.pad(tensor, paddings)
    return result


def pad_candidates(candidates):
    xy = [candidate.size() for candidate in candidates]
    xs = (x for x, _ in xy)
    ys = (y for _, y in xy)
    x, y = tuple(max(d) for d in (xs, ys))
    return torch.stack([pad(tensor, x, y) for tensor in candidates])


class ClozeQADataset(Dataset):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def collate_examples(examples: List[Example], train=True):
    fields = ('input_ids', 'attention_mask', 'token_type_ids',
              'p_mask', 'candidates')

    fields = fields + ('start_position', ) if train else fields

    values = {
        field: [example.__getattribute__(field) for example in examples]
        for field in fields
    }

    padded_values = {
        field: (
            pad_candidates(data) if field == 'candidates'
            else pad_sequence(data, batch_first=True)
        )
        for field, data in values.items()
    }

    return padded_values


def evaluate(model, data):
    sum_accuracy = 0
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collate_examples)
    for batch in tqdm(dataloader, desc='Evaluating'):
        model.eval()
        batch = {field: t.to(device) for field, t in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, train=False)
            p, *_ = outputs
            sum_accuracy += \
                torch.sum(torch.argmax(p, dim=1) == torch.argmax(batch['start_position'], dim=1)).item()

    return sum_accuracy / len(data)


def train(model, data):
    tb_writer = SummaryWriter()

    train_dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collate_examples)

    num_train_epochs = num_train_epochs_set
    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    max_accuracy = 0.0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(num_train_epochs, desc='Epoch')
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = {field: t.to(device) for field, t in batch.items()}
            outputs = model(**batch)
            loss, *_ = outputs
            loss /= gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if logging_steps > 0 and global_step % logging_steps == 0:
                    tb_writer.add_scalar('Train/Loss', (tr_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = tr_loss

                if global_step == 1 or eval_steps > 0 and global_step % eval_steps == 0:
                    accuracy = evaluate(model, valid_data)
                    tb_writer.add_scalar('Eval/Accuracy', accuracy, global_step)
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        logger.info('Saving model...')
                        model.save_pretrained(MODEL)
                    tb_writer.add_scalar('Eval/MaxAccuracy', max_accuracy, global_step)
                    logger.info(f"EVAL Accuracy: {accuracy} Max accuracy: {max_accuracy}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true',
                        help='Whether to do training')
    parser.add_argument('--eval', action='store_true',
                        help='Whether to do evaluation')
    parser.add_argument('--model', type=str, default=str(BERT),
                        help='Path to model')
    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%d/%m/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)

    model = BertForClozeQA.from_pretrained(args.model)
    model.to(device)

    if args.train:
        logger.info('Loading preprocessed data...')
        train_data = ClozeQADataset(torch.load(CACHE/'train.pt'))
        valid_data = ClozeQADataset(torch.load(CACHE/'valid.pt'))

        logger.info('Start training...')
        train(model, train_data)

    if args.eval:
        logger.info('Loading preprocessed data...')
        test_data = ClozeQADataset(torch.load(CACHE/'test.pt'))

        logger.info('Start evaluating...')
        accuracy = evaluate(model, test_data)
        logger.info(f"TEST Accurracy: {accuracy}")


if __name__ == '__main__':
    main()
