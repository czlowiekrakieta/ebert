import re

import numpy as np

def _preprocess(title):
    _replace = [(' is', "'s"), ('&', 'and')]
    _get_rid_of = [',', '.', "'", ':', '!', '?', ';', '(', ')']
    _cut_at_the_end = [', a', ', the']

    title = title.lower()
    for old, new in _replace:
        title = title.replace(old, new)

    for sign in _get_rid_of:
        title = title.replace(sign, '')

    for seq in _cut_at_the_end:
        if title.endswith(seq):
            title = title.replace(seq, '')

    return title

