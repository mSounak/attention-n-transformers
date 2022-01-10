import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Input, Embedding, Dropout, Dense, LayerNormalization
from positional_encoding import PositionalEncoding
from transformer_decoder import Decoder
from transformer_encoder import Encoder


# Parameters
embed_dim = 256
laten_dim = 2048
num_heads = 8
vocab_size = 10000
enc_seq_len = 30
dec_seq_len = 29

# Encoder
enc_input = Input(shape=(None,), name='enc_input')
enc_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)(enc_input)
enc_pos_encoding = PositionalEncoding(max_seq_len=enc_seq_len, embed_dim=embed_dim)(enc_embedding)
enc_out = Encoder(embed_dim, laten_dim, num_heads)(enc_pos_encoding)
encoder = tf.keras.Model(enc_input, enc_out)

# Decoder
dec_input = Input(shape=(None,), name = 'dec_input')
enc_seq_input = Input(shape=(None, embed_dim), name = 'enc_seq_input')
dec_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)(dec_input)
dec_pos_encoding = PositionalEncoding(max_seq_len=dec_seq_len, embed_dim=embed_dim)(dec_embedding)
dec_out = Decoder(embed_dim, laten_dim, num_heads)(dec_pos_encoding, enc_seq_input)
drop_out = Dropout(0.5)(dec_out)
out = Dense(vocab_size, activation='softmax')(drop_out)
decoder = tf.keras.Model([dec_input, enc_seq_input], out)

decoder_output = decoder([dec_input, enc_out])

transformer = tf.keras.Model(inputs=[enc_input, dec_input], outputs=decoder_output, 
name='transformers')







