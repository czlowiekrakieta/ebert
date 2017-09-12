import os
import argparse
import sys
from random import sample

import pandas as pd
import numpy as np
import tensorflow as tf

from ebert import config as cfg
from ebert.models.sequences import build_rnn
from ebert.trainers.helpers import _pad, concat_with_zeros, sample_cap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def fetch_data(cutoff=1, small=False, val_split=.8):

    base = cfg.MOVIELENS if not small else cfg.MOVIELENS_SMALL
    df = pd.read_csv(os.path.join(base, 'ratings.csv'))
    
    movie_titles = pd.read_csv(os.path.join(base, 'movies.csv'))

    if cutoff > 0:
        counts = df.movieId.value_counts()
        counts = counts[counts>cutoff]
        print('use cutoff')
        df = df.set_index('movieId').loc[counts.index, :].reset_index().sort_values('userId')

    uniq_mov = df.movieId.unique()
    mapping = dict(zip(uniq_mov.tolist(), range(uniq_mov.shape[0])))
    titlemap = dict(zip(movie_titles.movieId, movie_titles.title))

    return df, mapping, {v: titlemap[k] for k, v in mapping.items()}


def train(args):

    sess = tf.Session()

    df, movie_mapping, title_mapping = fetch_data(args.cutoff, args.small=='true')
    print('fetched data')
    uniq_users = df.userId.unique()

    train_data = df[df.timestamp < df.timestamp.quantile(args.train_cutoff)]
    val_data = df[df.timestamp > df.timestamp.quantile(args.train_cutoff)]

    val_available_users = set(val_data.userId.unique()) & set(train_data.userId.unique())
    val_available_users = list(val_available_users)

    print(len(val_available_users), ' users available for validation')

    train_data.set_index('userId', inplace=True)
    val_data.set_index('userId', inplace=True)

    if args.user_aware == 'true':
        raise NotImplementedError

    bag = build_rnn(len(movie_mapping),
                    cell_type=args.cell_type,
                    user_aware=args.user_aware == 'true',
                    user_cnt=uniq_users.shape[0],
                    rating_aware=args.ratings_aware == 'true',
                    movie_emb_dim=args.movie_emb_dim,
                    feed_previous=args.feed_prev == 'true',
                    rnn_unit=args.rnn_unit,
                    user_embedding=args.user_embed_dim)
    print('built rnn')
    print(len(movie_mapping), df.movieId.max())
    saver = tf.train.Saver(var_list=[x for x in tf.trainable_variables() if 'embedding' in x.name])

    movie_idx, total_loss, clf_loss = bag['base']
    if args.ratings_aware == 'true':
        true_ratings, rating_loss = bag['ratings']

    opt = tf.train.MomentumOptimizer(1e-3, momentum=.9).minimize(total_loss)

    idx = 0

    sess.run(tf.global_variables_initializer())

    validation_losses = []

    while idx < args.iters:

        users = sample(uniq_users.tolist(), args.batch_size)

        def listify(y, with_reversal=False):
            return list(map(lambda x: x.tolist()[::-1] if with_reversal else x.tolist(), y))

        def get_data(df, users, rev=False):
            mov = df.loc[users, ['movieId', 'rating']]
            temp_mov_idx = []
            temp_rats = []
            notfuckups = []

            for i, us in enumerate(users):
                temp = mov.loc[us, :]
                if len(temp.shape) != 2:
                    continue
                temp_mov_idx.append(temp['movieId'].map(movie_mapping).values)
                temp_rats.append(temp['rating'].values)
                notfuckups.append(i)

            temp_mov_idx = _pad(listify(temp_mov_idx, rev), 0)
            temp_rats = _pad(listify(temp_rats, rev), 0)
            return temp_mov_idx, temp_rats, notfuckups

        temp_mov_idx, temp_rats, train_idx = get_data(train_data, users)
        feed_dict = {movie_idx: temp_mov_idx[:, :args.maxlen], bag['feed_prev']:[True]*args.maxlen}
        fetches = [opt, total_loss, clf_loss]

        if args.ratings_aware == 'true':
            feed_dict.update({true_ratings: temp_rats[:, 1:args.maxlen]})
            # we don't predict rating for first movie
            fetches.append(rating_loss)

        feed_dict = {k: concat_with_zeros(v, args.batch_size)
                    if isinstance(v, np.ndarray) else v
                     for k, v in feed_dict.items()}
        res = sess.run(fetches, feed_dict=feed_dict)

        if idx % 100 == 0:
            print('training losses:')
            print('total loss\t', res[1])
            print('classifier loss\t', res[2])
            if args.ratings_aware == 'true':
                print('rating loss\t', res[3])

            val_users = sample_cap(val_available_users, 4*args.batch_size)
            pre_val_mov_idx, pre_val_rats, train_ok = get_data(train_data, val_users, True)
            past_val_mov_idx, past_val_rats, val_ok = get_data(val_data, val_users, False)

            half = args.maxlen // 2

            val_mov_idx = np.concatenate((pre_val_mov_idx[:, -half:], past_val_mov_idx[:, :half]), axis=1)
            val_rat = np.concatenate((pre_val_rats[:, -half:], past_val_rats[:, :half]), axis=1)[:, 1:]
            fetches = [total_loss, clf_loss]

            feed_dict = {movie_idx:val_mov_idx,
                         bag['feed_prev']: [False]*half + [True]*half,
                         bag['if_training']: False,
                         }
            if args.ratings_aware == 'true':
                feed_dict.update({true_ratings: val_rat})
                fetches.append(rating_loss)

            feed_dict = {k: concat_with_zeros(v, args.batch_size)
                        if isinstance(v, np.ndarray) else v
                         for k, v in feed_dict.items()}
            res = sess.run(
                fetches=fetches,
                feed_dict=feed_dict
            )

            print('validation losses:')
            print('total loss:\t', res[0])
            print('classfier loss:\t', res[1])
            print('ratings loss:\t', res[2])

            better = np.any(res[0] < np.asarray(validation_losses[-5:]))
            if args.save_model == 'true' and idx > 500 and better:
                
                if not os.path.exists('embeddings'):
                    os.mkdir('embeddings')
                
                movie_embeddings = sess.run(bag['movie_embeddings']).astype(np.float16)
                import pickle
                
                with open(os.path.join('embeddings', args.model_name), 'rb') as f:
                    pickle.dump({'movie_embeddings': movie_embeddings, 'mapping': title_mapping}, f)
                                           
        idx += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    def add_arg(name, default=None, type=str, req=False, parser=parser, choices=None):
        parser.add_argument('--' + name, default=default, type=type, required=req, choices=choices)

    add_arg('cell_type', default='gru')
    add_arg('user_aware', default='false')
    add_arg('ratings_aware', default='true')
    add_arg('feed_prev', default='true')
    add_arg('rnn_unit', default=300, type=int)
    add_arg('movie_emb_dim', default=300, type=int)
    add_arg('small', default='true')
    add_arg('user_embed_dim', default='true')
    add_arg('batch_size', default=32, type=int)
    add_arg('iters', default=10000, type=int)
    add_arg('cutoff', default=0, type=int)
    add_arg('maxlen', default=200, type=int)
    add_arg('train_cutoff', default=.8, type=float)
    add_arg('model_name')
    add_arg('save_model', default='false', choices=['false', 'true'])

    args = parser.parse_args()

    train(args)