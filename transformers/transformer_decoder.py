import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Embedding, Dropout, Dense, LayerNormalization, MultiHeadAttention

# Transformer Decoder

class Decoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention_1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        self.layer_norm3 = LayerNormalization()

        self.fc = tf.keras.Sequential([
            Dense(dense_dim, activation='relu'),
            Dense(embed_dim)
        ])

    def call(self, dec_input, enc_output, look_ahead_mask, padding_mask):

        attn_out1 = self.attention_1(
            query=dec_input, value=dec_input, key=dec_input, attention_mask=look_ahead_mask)

        x1 = self.layer_norm1(dec_input + attn_out1)

        attn_out2 = self.attention_2(
            query=x1, value=enc_output, key=enc_output, attention_mask=padding_mask)

        x2 = self.layer_norm2(x1 + attn_out2)

        fc_out = self.fc(x2)
        
        x3 = self.layer_norm3(x2 + fc_out)

        return x3