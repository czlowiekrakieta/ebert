import os
import re
import json
from collections import defaultdict

import pandas as pd

from ebert import config as cfg

dd_mm_rrrr = re.compile(r'\d{2}-\d{2}-\d{4}')
rrrr_mm_dd = re.compile(r'\d{4}-\d{2}-\d{2}')


def _path(filename):
    return os.path.join(cfg.SUMMARIES, filename)


def _undict_genres(value):
    return json.loads(value).values()


def _read_decade(year):
    if dd_mm_rrrr.match(year):
        year = year.split('-')[-1]
    elif rrrr_mm_dd.match(year):
        year = year.split('-')[0]

    year = max(int(year), 1890)
    return (year - 1890) // 10


def read_plots(target='genre', with_languages=False, with_production_country=False):
    """
    reads plots with label assigned by user

    :param target: can be 'genre' or 'decade'
    :return: plots, labels, titles, ids
    """

    title_col = 2
    id_col = 1

    if target not in ['genre', 'decade']:
        raise ValueError('you can build models for genre, '
                         'or decade of production')

    info = pd.read_csv(_path('movie.metadata.tsv'),
                       header=None, sep='\t').set_index(0)
    with open(_path('plot_summaries.txt'), 'r') as f:
        plots = f.read().split('\n')[:-1]
        idx, plots = list(zip(*map(lambda x: x.split('\t'), plots)))
        idx = list(map(int, idx))

    print('read plots')

    if target == 'genre':
        target_col = 8
        func = _undict_genres

    elif target == 'decade':
        target_col = 3
        func = _read_decade

    labels = info[target_col].apply(func)
    labels = labels.loc[idx].apply(
        lambda x: [] if pd.isnull(x) else list(x)
    )

    return plots, labels, info[title_col].loc[idx], info[id_col].loc[idx]
