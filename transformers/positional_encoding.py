import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Embedding, Dropout, Dense, LayerNormalization

# Positional Encoding

class PositionalEncoding(Layer):
    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.dropout = Dropout(0.1)

    def call(self, x):
        pe = np.array([get_angles(pos) for pos in range(self.max_seq_len)])
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        seq_len = tf.shape(x)[1]
        x = x * tf.math.sqrt(tf.cast(self.embed_dim), dtype=tf.float32)

        x = x + pe[:, :seq_len, :]

        x = self.dropout(x)

        return x

    
    def get_angles(self, pos):
        angle_rates = [pos / np.power(10000, (2 * (i // 2)) / np.float32(self.embed_dim))
            for i in range(self.embed_dim)]
        
        return angle_rates