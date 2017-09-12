import argparse
import tensorflow as tf
import numpy as np
import os
import sys

from ebert.models.classification.bags import bow_model
from ebert.models.classification.rnns import rnn_model
from ebert.models.classification.general import build_loss_and_training
from ebert.readers.plots import read_plots
from ebert.trainers.helpers import tokenize_seqs, _build_binary_mask, _melt, _pad, presplit
from ebert.logger import Logger

from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_graph(args):

    plots, labels = read_plots(args.target, args.langs == 'true', args.country == 'true')[:2]
    plots_train, plots_valid, labels_train, labels_valid = train_test_split(plots, labels)
    plots_train, plots_valid = list(map(presplit,
                                        [plots_train, plots_valid]))
    plots_train_tok, plots_val_tok, idx = tokenize_seqs(sequences_train=plots_train,
                                                        sequences_val=plots_valid,
                                                        count_threshold=args.counts_threshold)

    labels_train, labels_valid, labels_idx = tokenize_seqs(sequences_val=labels_valid,
                                                           sequences_train=labels_train,
                                                           count_threshold=0)
    if len(plots_train_tok) != len(labels_train):
        raise ValueError

    print('training samples: {} \nvalidation samples: {}'.format(
        len(plots_train_tok), len(plots_val_tok)
    ))

    # tutaj jakies if args.model_type == 'mlp'
    # alternatywy: rnn i cnn
    # dla kazdego osobny zestaw paramtrow, dla ktorego reszta bedzie bez znaczenia

    if args.model_type == 'bag':
        model_output = bow_model(vocab_size=len(idx),
                                 embed_dim=args.embed_dim,
                                 objs_per_sample=args.max_len,
                                 num_classes=len(labels_idx),
                                 hidden_layer_sizes=args.hidden,
                                 activations=args.activations,
                                 dropouts=args.dps,
                                 bns=[x == 'true' for x in args.bns],
                                 attention=args.attention == 'true')

    elif args.model_type == 'rnn':
        model_output = rnn_model(vocab_size=len(idx),
                                 embed_dim=args.embed_dim,
                                 objs_per_sample=args.max_len,
                                 num_classes=len(labels_idx),
                                 rnn_units=args.rnn_units,
                                 bidir=args.bidir == 'bidir',
                                 cell_type=args.cell_type,
                                 hidden_layer_sizes=args.hidden,
                                 activations=args.activations,
                                 dropouts=args.dps,
                                 take_mean=args.take_mean == 'true',
                                 bns=[x == 'true' for x in args.bns])

    print('built model')

    loss, opt, labels_ph = build_loss_and_training(model_output[0][-1],
                                                   len(labels_idx),
                                                   args.learning_rate)
    attn = args.model_type != 'bag' or args.attention == 'true'
    if attn:
        model_flesh, input_ph, embedding, is_training = model_output
    else:
        model_flesh, input_ph, embedding, is_training, mask = model_output

    bag =  (input_ph, labels_ph, loss, opt,
            plots_train_tok, plots_val_tok,
            labels_train, labels_valid,
            len(labels_idx), len(idx), is_training)

    if not attn:
        bag += (mask, )

    return bag


def train(args):

    attn = args.model_type != 'bag' or args.attention == 'true'

    bag = build_graph(args)
    if attn:
        (input_ph, labels_ph, loss, opt,
         plots_train_tok, plots_val_tok,
         labels_train, labels_valid,
         num_classes, num_tokens, state) = bag
    else:
        (input_ph, labels_ph, loss, opt,
         plots_train_tok, plots_val_tok,
         labels_train, labels_valid,
         num_classes, num_tokens, state,
         mask) = bag

    def prepare_batch(x, y):
        x_padded = _pad(x)
        y_lab = np.zeros((len(y), num_classes))
        y_lab[_melt(y)] += 1.
        feed_dict = {input_ph: x_padded, labels_ph: y_lab}
        return feed_dict

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for e in range(args.epochs):
        offset = 0

        while offset < len(plots_train_tok):
            x = plots_train_tok[offset:offset+args.batch_size]
            y = labels_train[offset:offset+args.batch_size]
            feed_dict = prepare_batch(x, y)

            if args.model_type == 'bag' and args.attention != 'true':
                mask_np = _build_binary_mask(list(map(len, x)))
                feed_dict.update({mask: mask_np})

            _, L = sess.run([opt, loss], feed_dict=feed_dict)
            offset += args.batch_size
            if offset / args.batch_size % 10 == 0:
                print(args.model_name,
                      '\tbatches: {} from {}'.format(
                          offset // args.batch_size,
                          len(plots_train_tok) // args.batch_size),
                      L)

        val_offset = 0
        vl = []
        while val_offset < len(plots_val_tok):
            x = plots_val_tok[val_offset:val_offset+args.batch_size]
            y = labels_valid[val_offset:val_offset+args.batch_size]
            feed_dict = prepare_batch(x, y)
            L = sess.run(loss, feed_dict=feed_dict)
            vl.append(L)
            val_offset += args.batch_size

        print('validation loss:,\t', np.mean(vl), np.std(vl))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--rnn_units', type=int, default=300)
    parser.add_argument('--bidir', type=str, choices=['false', 'true'])
    parser.add_argument('--hidden', type=int, default=(100, 100), nargs='*')
    parser.add_argument('--activations', type=int, default=('relu', 'relu'), nargs='+')
    parser.add_argument('--dps', type=float, default=(0, 0), nargs='+')
    parser.add_argument('--attention', type=str, choices=['true', 'false'], default='false')
    parser.add_argument('--bns', type=str, default=['false', 'false'], nargs='+')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--target', type=str, default='genre', choices=['genre'])
    parser.add_argument('--langs', type=str, default='false')
    parser.add_argument('--cell_type', type=str, default='gru')
    parser.add_argument('--country', type=str, default='false')
    parser.add_argument('--counts_threshold', type=int, default=15)
    parser.add_argument('--model_name', type=str, default='nn_on_simple_bow')
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--take_mean', type=str, default='true')
    parser.add_argument('--log', type=str, default='true', choices=['false', 'true'])

    args = parser.parse_args()

    if args.log != 'false':
        if not os.path.exists('logs'):
            os.mkdir('logs')

        sys.stdout = Logger('logs/' + args.model_name)

    print(args)

    train(args)
