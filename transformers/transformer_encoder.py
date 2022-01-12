import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Embedding, Dropout, Dense, LayerNormalization, MultiHeadAttention


# Transformer Encoder block

class Encoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()

        self.fc = tf.keras.Sequential([
            Dense(dense_dim, activation='relu'),
            Dense(embed_dim)
        ])
        

    def call(self, inputs, padding_mask):

        attn_out = self.attention(query=inputs, value=inputs, key=inputs,
        attention_mask=padding_mask)

        x1 = self.layer_norm1(inputs + attn_out)

        fc_out = self.fc(x1)

        x2 = self.layer_norm2(x1 + fc_out)

        return x2