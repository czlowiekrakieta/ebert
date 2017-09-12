import os
import re

import pandas as pd
import numpy as np

from ebert import config as cfg

yr = re.compile(r'(\d{4})')


def _clear_year(text):
    no_year = yr.sub('', text)
    year = yr.findall(text)
    return year, no_year


def _path(filename):
    return os.path.join(cfg.MOVIELENS, filename)


def _read_movies():
    movies = pd.read_csv(_path('movies.csv'))
    return pd.concat([
        pd.DataFrame(movies.title.apply(_clear_year).tolist(),
                     columns=['year', 'title']),
        movies[['movieId', 'genres']]],
        axis=1)


def read_sequences(with_ratings=False):
    ratings = pd.read_csv(_path('ratings.csv'))


def read_tags(relevance_threshold=.1, movie_subset=None):
    movies_with_tags = pd.read_csv(_path('genome-scores.csv')).set_index('movieId')

    tags = {}
    ids_to_title = {}

    movies = _read_movies().set_index('movieId')
    if movie_subset is None:
        movie_subset = np.unique(movies['movieId'].values)

    for mov in movie_subset:
        sub = movies_with_tags.loc[mov]
        tags[movies.loc[mov, 'title']] = {
            'tags': sub['tagId'][sub['relevance'] > relevance_threshold],
            'genres': movies.loc[mov, 'genres'],
            'year': movies.loc[mov, 'year']
        }
        ids_to_title[mov] = movies.loc[mov, 'title']

    return tags, ids_to_title
