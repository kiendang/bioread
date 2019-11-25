SEPARATOR = '------------------------------'


def make_entity_dict(s):
    lines = s.strip().split('\n')
    lines = (line.strip() for line in lines)
    pairs = [line.split(':')[:2] for line in lines]
    return pairs


def multi_replace(s, replacements):
    for r in replacements:
        s = s.replace(*r)
    return s


def split_instance(text, separator=SEPARATOR):
    context, question, entities, answer = text.split(SEPARATOR)[:4]
    return (context, question, entities, answer)
