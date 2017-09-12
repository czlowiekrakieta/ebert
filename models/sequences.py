import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, LSTMStateTuple
from tensorflow.contrib.layers import linear, xavier_initializer, fully_connected

cells = {
    'lstm': LSTMCell,
    'gru': GRUCell
}

def build_embedding(vocab, dim, name='embedding'):
    with tf.variable_scope(initializer=xavier_initializer(), name_or_scope=name):
        return tf.get_variable(name=name, shape=[vocab, dim])


def build_rnn(movies_cnt, cell_type='gru', user_aware=True, user_cnt=None, rating_aware=True,
              rnn_unit=300, user_embedding=300, movie_emb_dim=300, feed_previous=True,
              loss_weights=None, rating_with_user=False, batch_size=32):

    loss_weights = loss_weights or [10, 2]
    movie_idx_ph = tf.placeholder(tf.int32, [None, None])
    _, maxlen = tf.unstack(tf.shape(movie_idx_ph))

    if_training = tf.placeholder_with_default(True, [])
    cell = cells[cell_type](num_units=rnn_unit)

    movie_embeddings = build_embedding(movies_cnt, movie_emb_dim, 'movie_embedding')

    if user_aware and user_cnt is None:
        raise ValueError

    if user_aware:

        user_idx_ph = tf.placeholder(tf.int32, [None])
        if cell_type == 'lstm':

            c_user_embedding = build_embedding(user_cnt, user_embedding, name='user_c_embedding')
            h_user_embedding = build_embedding(user_cnt, user_embedding, name='user_h_embedding')
            state = LSTMStateTuple(c=tf.nn.embedding_lookup(c_user_embedding, user_idx_ph),
                                   h=tf.nn.embedding_lookup(h_user_embedding, user_idx_ph))

        elif cell_type == 'gru':

            user_embedding = build_embedding(user_cnt, user_embedding, name='user_embedding')
            state = tf.nn.embedding_lookup(user_embedding, user_idx_ph)

    else:

        state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

    def _choose_best(vec, reuse=False):
        with tf.variable_scope(name_or_scope='chooser', reuse=reuse) as scope:
            w = tf.get_variable(name='weights', shape=[movie_emb_dim, movies_cnt])
            b = tf.get_variable(name='bias', shape=[movies_cnt])
            return tf.matmul(vec, w) + b

    # not using dynamic_rnn since I want to feed previous output

    def walker(idx, input, outputs, state, fprev):

        output, state = cell(input, state)

        new_idx = tf.cond(fprev[idx],
                          lambda: tf.cast(tf.argmax(_choose_best(output), 1), tf.int32),
                          lambda: movie_idx_ph[:, idx+1])

        input = tf.nn.embedding_lookup(movie_embeddings, new_idx)

        return idx+1, input, tf.concat((outputs, tf.expand_dims(output, axis=1)), axis=1), state, fprev

    def cond(idx, input, outputs, state, fprev):
        return idx < maxlen - 1

    idx = tf.Variable(0)
    input = tf.nn.embedding_lookup(movie_embeddings, movie_idx_ph[:, 0])

    feed_prev = tf.placeholder(tf.bool, [None], name='feed_prev_ph')

    loop_vars = [idx,
                 input,
                 tf.zeros((batch_size, 0, movie_emb_dim), dtype=tf.float32),
                 state,
                 feed_prev]

    shape_invs = [idx.get_shape(),
                  input.get_shape(),
                  tf.TensorShape((batch_size, None, movie_emb_dim)),
                  state.get_shape(),
                  feed_prev.get_shape()]

    print(len(loop_vars), len(shape_invs))
    idx, last_output, outputs, state, fp = tf.while_loop(cond,
                                                         walker,
                                                         loop_vars=loop_vars,
                                                         shape_invariants=shape_invs)

    logits = tf.reshape(outputs, (-1, rnn_unit))
    logits = _choose_best(logits, reuse=True)
    logits = tf.reshape(logits, (batch_size, -1, movies_cnt))

    clf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=movie_idx_ph[:, 1:])

    def training_mask():
        clf_mask = tf.greater(movie_idx_ph[:, 1:], tf.cast(0, dtype=tf.int32))
        clf_mask = tf.cast(clf_mask, tf.float32)
        return clf_mask

    def val_mask():
        clf_mask = tf.greater(movie_idx_ph[:, 1:], tf.cast(0, dtype=tf.int32))
        clf_mask = tf.cast(clf_mask, tf.float32)
        clf_mask = tf.multiply(clf_mask, tf.cast(feed_prev[1:], tf.float32))
        return clf_mask

    clf_mask = tf.cond(if_training, training_mask, val_mask)
    clf_loss *= clf_mask
    clf_loss = tf.reduce_sum(clf_loss) / tf.reduce_sum(clf_mask)
    total_loss = loss_weights[0]*clf_loss

    if rating_aware:
        true_ratings = tf.placeholder('float', [None, None])

        ratings = linear(outputs, 1)
        ratings = tf.squeeze(ratings, axis=2)

        rat_loss = tf.square(ratings - true_ratings)
        mask = tf.greater(true_ratings, tf.cast(0, dtype=tf.float32))
        mask = tf.cast(mask, tf.float32)
        rat_loss *= mask
        rat_loss = tf.reduce_sum(rat_loss) / tf.reduce_sum(mask)
        total_loss += loss_weights[1]*rat_loss

    bag = {
        'base': [movie_idx_ph, total_loss, clf_loss],
        'feed_prev': feed_prev,
        'if_training': if_training,
        'movie_embeddings': movie_embeddings
    }

    if user_aware:
        bag['user'] = user_idx_ph

    if rating_aware:
        bag['ratings'] = [true_ratings, rat_loss]

    return bag
