import tensorflow as tf


def create_padding_mask(inputs):
    inputs = tf.cast(tf.math.not_equal(inputs, 0), tf.float32)

    return inputs[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)