import numpy as np
import torch

from more_itertools import grouper, windowed

from dataclasses import dataclass
from functools import reduce
from itertools import accumulate, chain, groupby
from operator import add, sub, itemgetter
from pathlib import Path


def split_instance(text):
    url, context, question, answer, entities = text.split('\n\n')
    return url, context, question, answer, entities


def make_entity_dict(s):
    lines = s.strip().split('\n')
    lines = (line.strip() for line in lines)
    pairs = (line.split(':')[:2] for line in lines)
    return dict(pairs)


def list_locate(l, x):
    return [i for i, e in enumerate(l) if e == x]


def indicator(l, x, p=1, n=0):
    assert len(x) <= l
    result = [n] * l
    for i in x:
        result[i] = p
    return result


def index_default(l, value, default=None):
    try:
        return l.index(value)
    except ValueError:
        return default if default is not None else len(l)


def window_split(seq, n, step, fillvalue, suffix=None, postprocess=None):
    windows = windowed(seq, n, step=step, fillvalue=fillvalue)
    windows = [list(window) for window in windows]

    if suffix is not None:
        for window in windows:
            window.insert(index_default(window, fillvalue), suffix)

    windows = (
        [postprocess(window) for window in windows]
        if postprocess is not None else windows
    )

    return windows


def score(i, l):
    return min(i, l - i - 1) + 0.01 * l


@dataclass
class Example:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    p_mask: torch.Tensor
    candidates: torch.Tensor
    start_position: torch.Tensor


def preprocess(text, tokenizer,
               max_seq_length=384,
               doc_stride=128,
               min_context_length=1,
               max_query_length=None):
    num_special_tokens = 3
    max_query_length = (
        min(max_query_length, max_seq_length - num_special_tokens - min_context_length)
        if max_query_length else max_seq_length - num_special_tokens - min_context_length
    )

    url, context, question, answer, entities = split_instance(text)
    entities_dict = make_entity_dict(entities)
    n_entities = len(entities_dict)

    context_words = context.split()

    context_words_unmasked = [entities_dict.get(word, word) for word in context_words]

    entity_positions = {
        entity: list_locate(context_words, entity)
        for entity, _ in entities_dict.items()
    }

    tokenized_words = [tokenizer.tokenize(word) for word in context_words_unmasked]
    word_positions = list(chain([0], accumulate(len(tokens) for tokens in tokenized_words)))

    context_tokens = list(chain.from_iterable(tokenized_words))

    entity_token_positions = {
        entity: [word_positions[position] for position in positions]
        for entity, positions in entity_positions.items()
    }

    question_words = question.split()
    question_words_unmasked = [
        {**entities_dict, **{'@placeholder': tokenizer.mask_token}}.get(word, word)
        for word in question_words
    ]
    question_tokens = tokenizer.tokenize(' '.join(question_words_unmasked))
    question_tokens = question_tokens[:max_query_length]

    query_length = len(question_tokens)
    context_segment_length = max_seq_length - query_length - num_special_tokens

    context_length = len(context_tokens)

    entity_token_indicator = {
        entity: indicator(context_length, positions)
        for entity, positions in entity_token_positions.items()
    }

    segment_ids = np.array(window_split(
        [1] * context_length, context_segment_length, doc_stride,
        0, 1, lambda x: [0] * (query_length + 2) + x
    ))

    attention_mask = np.array(window_split(
        [1] * context_length, context_segment_length, doc_stride,
        0, 1, lambda x: [1] * (query_length + 2) + x
    ))

    tokens_windowed = window_split(
        context_tokens, context_segment_length, doc_stride,
        tokenizer.pad_token, tokenizer.sep_token,
        lambda x: [tokenizer.cls_token] + question_tokens + [tokenizer.sep_token] + x
    )

    token_ids = np.array([
        tokenizer.convert_tokens_to_ids(tokens)
        for tokens in tokens_windowed
    ])

    segment_lengths = tuple(
        - sub(*list_locate(l, tokenizer.sep_token)[:2]) - 1
        for l in tokens_windowed
    )
    n_segments = len(segment_lengths)

    offsets = (0, ) + tuple(accumulate(segment_lengths))[:-1]

    token_idx = range(context_length)
    token_idx_windowed = windowed(token_idx, context_segment_length, step=doc_stride)

    token_idx_scored = chain.from_iterable(
        ((x, i + offset, score(i, l)) for i, x in enumerate(xs) if x is not None)
        for xs, l, offset
        in zip(token_idx_windowed, segment_lengths, offsets)
    )

    max_context = [i for _, i, _ in (
        max(x, key=itemgetter(2))
        for _, x
        in groupby(sorted(token_idx_scored, key=itemgetter(0)), key=itemgetter(0))
    )]

    max_context_indicator = indicator(sum(segment_lengths), max_context)

    max_context_windowed = window_split(
        max_context_indicator, context_segment_length, context_segment_length,
        0, postprocess=lambda x: [0] * (query_length + 2) + x + [0]
    )

    max_context_mask = np.asarray(max_context_windowed)

    entities_mask = {
        entity: np.asarray(window_split(
            indi, context_segment_length, doc_stride, 0,
            postprocess=lambda x: [0] * (query_length + 2) + x + [0]
        ))
        for entity, indi in entity_token_indicator.items()
    }

    entities_max_context_mask = {
        entity: mask * max_context_mask
        for entity, mask in entities_mask.items()
    }

    p_mask = sum(mask for _, mask in entities_max_context_mask.items())
    p_mask_flat = p_mask.flatten()

    entities_max_context_mask_pairs = tuple(entities_max_context_mask.items())
    entities_list = [entity for entity, _ in entities_max_context_mask_pairs]
    start_position = entities_list.index(answer.strip())
    start_position_indicator = np.array(indicator(len(entities_list), [start_position]))

    candidates = np.asarray([mask.flatten() for _, mask in entities_max_context_mask_pairs])

    result = Example(**{
        name: torch.tensor(array)
        for name, array in (
            ('input_ids', token_ids),
            ('attention_mask', attention_mask),
            ('token_type_ids', segment_ids),
            ('p_mask', p_mask_flat),
            ('candidates', candidates),
            ('start_position', start_position_indicator)
        )
    })

#     for x, i in entities_mask.items():
#         assert len(i) == n_segments
#         assert len(i[0]) == max_seq_length

#     for x, i in entities_max_context_mask.items():
#         assert i.shape == (n_segments, max_seq_length)
#         assert np.sum(i) == len(entity_positions[x])

#     assert p_mask.shape == (n_segments, max_seq_length)
#     assert np.sum(p_mask[:, :(query_length + 2)]) == 0
#     assert np.sum(p_mask) == sum(len(pos) for _, pos in entity_positions.items())
#     assert p_mask_flat.shape == (n_segments * max_seq_length, )
#     assert candidates.shape == (n_entities, n_segments * max_seq_length)
#     assert token_ids.shape == (n_segments, max_seq_length)
#     assert segment_ids.shape == (n_segments, max_seq_length)
#     assert np.sum(segment_ids) == sum(segment_lengths) + n_segments
#     assert attention_mask.shape == (n_segments, max_seq_length)
#     assert np.sum(attention_mask) == sum(segment_lengths) + n_segments * (query_length + 3)
#     assert sum(start_position_indicator) == 1
#     assert start_position_indicator.shape == (n_entities, )

    return result
