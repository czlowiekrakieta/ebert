from functools import partial

import tensorflow as tf

from tensorflow.contrib.rnn import (LSTMCell,
                                    GRUCell,
                                    DropoutWrapper,
                                    LSTMStateTuple,
                                    MultiRNNCell)
from tensorflow.python.layers.core import dense

from ebert.models.classification.general import _protos_builder, _mlp


cells = {
    'lstm': LSTMCell,
    'gru': GRUCell,
    'multi': MultiRNNCell
}


def rnn_model(vocab_size, embed_dim, objs_per_sample,
              num_classes, cell_type, bidir, rnn_units,
              hidden_layer_sizes=(512, 1024, 1024),
              activations=('relu', 'relu', 'relu'),
              dropouts=(.5, .1, .1),
              bns=(False, False, False),
              take_mean=False,
              attention=False):

    cell = cells[cell_type](num_units=rnn_units)
    embedding, input_ph, vecs = _protos_builder(vocab_size=vocab_size, embed_dim=embed_dim)

    if not bidir:
        outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=vecs, dtype=tf.float32)
    else:
        ((outputs_fw, outputs_bw),
        (states_fw, states_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                                  cell_bw=cell,
                                                                  inputs=vecs,
                                                                  dtype=tf.float32)
        outputs = tf.concat((outputs_bw, outputs_fw), axis=2)

    output = tf.reduce_mean(outputs, axis=1) if take_mean else outputs[:, -1, :]

    is_training = tf.placeholder('float', [])

    if hidden_layer_sizes and not hidden_layer_sizes[0] == 0:
        layers = _mlp(output, is_training, hidden_layer_sizes, activations, dropouts, bns)
    else:
        layers = [output]

    layers.append(dense(layers[-1], num_classes))

    return layers, input_ph, embedding, is_training
