import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.layers.convolutional import conv2d
from tensorflow.python.layers.core import dense, dropout
from tensorflow.python.layers.normalization import batch_normalization

act_funcs = {
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid
}

optimizers = {
    'sgd': tf.train.GradientDescentOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'adam': tf.train.AdamOptimizer,
    'momentum': tf.train.MomentumOptimizer,
}


def _mlp(input,
         is_training,
         hidden_layer_sizes=(512, 1024, 1024),
         activations=('relu', 'relu', 'relu'),
         dropouts=(.5, .1, .1),
         bns=(False, False, False)):

    layers = [input]

    for hl, act, dp, bn in zip(
        hidden_layer_sizes, activations, dropouts, bns
    ):
        layers.append(dense(layers[-1], hl, activation=act_funcs[act]))
        if dp == 1.:
            layers.append(dropout(layers[-1], rate=dp, training=is_training))
        if bn:
            layers.append(batch_normalization(layers[-1], training=is_training))

    return layers


def _put_mask(vectors, mask, seq_lens=None):
    mult = tf.cast(tf.expand_dims(mask, 2), tf.float32) * vectors
    mult = tf.reduce_sum(mult, axis=1)
    if seq_lens is not None:
        mult = tf.div(mult, tf.cast(dtype=tf.float32, x=tf.expand_dims(seq_lens, 1)))
    return mult


def _protos_builder(vocab_size,
                    embed_dim,
                    objs_per_sample=None):
    embedding = build_embedding(vocab=vocab_size, dim=embed_dim)
    input_ph = tf.placeholder(tf.int32, [None, None], name='inputs_indices')
    vecs = tf.nn.embedding_lookup(embedding, input_ph)
    return embedding, input_ph, vecs


def _attend(vecs, attn_size):
    with tf.variable_scope(initializer=xavier_initializer(), name_or_scope='attention') as scope:
        inspector_vec = tf.get_variable('inspector', [attn_size])
        vec_length, vec_size = tf.unstack(tf.shape(vecs))[1:3]
        vecs = tf.reshape(vecs, [-1, vec_length, 1, vec_size])

        hidden_features = conv2d(vecs, attn_size,
                                 strides=1,
                                 kernel_size=1,
                                 padding='SAME',
                                 activation=tf.nn.relu)
        hidden_features = tf.reshape(hidden_features, [-1, vec_length, attn_size])

        s = tf.reduce_sum(
            hidden_features * inspector_vec, axis=-1
        )
        s = tf.nn.softmax(s, dim=-1)

        return s


def build_embedding(vocab, dim, name='embedding'):
    with tf.variable_scope(initializer=xavier_initializer(), name_or_scope=name):
        return tf.get_variable(name=name, shape=[vocab, dim])


def build_loss_and_training(logits, num_classes, learning_rate=1e-2, algo='sgd', **params):
    labels = tf.placeholder('float', [None, num_classes], name='labels_indices')
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss)

    opt = optimizers[algo](learning_rate=learning_rate).minimize(loss)
    return loss, opt, labels