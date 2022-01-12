import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Input, Embedding, Dropout, Dense, LayerNormalization
from positional_encoding import PositionalEncoding
from transformer_decoder import Decoder
from transformer_encoder import Encoder
from masking import create_look_ahead_mask, create_padding_mask


# Parameters
embed_dim = 256
laten_dim = 2048
num_heads = 8
vocab_size = 10000
enc_seq_len = 30
dec_seq_len = 30


# Encoder
enc_input = Input(shape=(enc_seq_len,), name='enc_input')

padding_mask = create_padding_mask(enc_input)

enc_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)(enc_input)
enc_pos_encoding = PositionalEncoding(max_seq_len=enc_seq_len, embed_dim=embed_dim)(enc_embedding)


enc_out = Encoder(embed_dim, laten_dim, num_heads)(enc_pos_encoding, padding_mask)

# Decoder
dec_input = Input(shape=(dec_seq_len, ), name = 'dec_input')

look_ahead_mask = create_look_ahead_mask(dec_input.shape[1])
dec_target_padding_mask = create_padding_mask(dec_input)
look_ahead_mask = tf.minimum(dec_target_padding_mask, look_ahead_mask)

dec_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)(dec_input)

dec_pos_encoding = PositionalEncoding(max_seq_len=dec_seq_len, embed_dim=embed_dim)(dec_embedding)

dec_out = Decoder(embed_dim, laten_dim, num_heads)(dec_pos_encoding, enc_out, look_ahead_mask, padding_mask)

drop_out = Dropout(0.5)(dec_out)

out = Dense(vocab_size, activation='softmax')(drop_out)

transformer = tf.keras.Model(inputs=[enc_input, dec_input], outputs=out, 
name='transformers')


e_inp = tf.random.uniform((1, enc_seq_len))
d_inp = tf.random.uniform((1, dec_seq_len))

out = transformer.predict([e_inp, d_inp])

print(out.shape)