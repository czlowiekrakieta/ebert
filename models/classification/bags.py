from functools import partial

import numpy as np
import tensorflow as tf

from ebert.models.classification.general import _attend, \
    _protos_builder, _put_mask, _mlp
from tensorflow.python.layers.core import dense


def bow_model(vocab_size,
             embed_dim,
             objs_per_sample,
             num_classes,
             hidden_layer_sizes=(512, 1024, 1024),
             activations=('relu', 'relu', 'relu'),
             dropouts=(.5, .1, .1),
             bns=(False, False, False),
             attention=False):

    if not isinstance(attention, (int, bool)):
        raise TypeError('attention has to be either False or 0, '
                        'which means that no attention will be applied, '
                        'or integer greater than zero - it will determine'
                        'its size.')

    embedding, input_ph, vecs = _protos_builder(vocab_size, embed_dim, objs_per_sample)
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    if not attention:
        mask = tf.placeholder('float', [None, None], name='mask')
    else:
        mask = _attend(vecs, attention)

    net = _put_mask(vecs, mask, tf.reduce_sum(mask, 1) if not attention else None)

    if len(hidden_layer_sizes):
        layers = _mlp(net, is_training, hidden_layer_sizes, activations, dropouts, bns)
    else:
        layers = [net]

    layers.append(dense(layers[-1], num_classes))

    if not attention:
        return layers, input_ph, embedding, is_training, mask

    return layers, input_ph, embedding, is_training

zoo = {
    'nn_on_simple_bow': partial(bow_model, attention=False),
    'nn_on_attentive_bow': partial(bow_model, attention=True)
}