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


# Model
class Transformer(tf.keras.Model):
    def __init__(self, embed_dim, laten_dim, num_heads, vocab_size, inp_seq_len, tar_seq_len, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.laten_dim = laten_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.inp_seq_len = inp_seq_len
        self.tar_seq_len = tar_seq_len

        self.encoder = Encoder(embed_dim, laten_dim, num_heads)
        self.decoder = Decoder(embed_dim, laten_dim, num_heads)

        self.pos_enc = PositionalEncoding(max_seq_len=inp_seq_len, embed_dim=embed_dim)
        self.pos_dec = PositionalEncoding(max_seq_len=tar_seq_len, embed_dim=embed_dim)

        self.enc_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.dec_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)

        self.dropout = Dropout(0.5)

    def call(self, inputs):
        enc_input, dec_input = inputs

        enc_padding_mask, dec_padding_mask, look_ahead_mask = self.compute_mask(enc_input, dec_input)
        
        print(f'Enc_padding: {enc_padding_mask.shape}')
        print(f'Dec_padding: {dec_padding_mask.shape}')
        print(f"Look_ahead: {look_ahead_mask.shape}")
        # Embeddings
        enc_seq_input = self.enc_embedding(enc_input)
        dec_seq_input = self.dec_embedding(dec_input)

        print(f'Enc_seq_input: {enc_seq_input.shape}')
        print(f'Dec_seq_input: {dec_seq_input.shape}')

        # Positional Encoding
        enc_seq_pos = self.pos_enc(enc_seq_input)
        dec_seq_pos = self.pos_dec(dec_seq_input)

        print(f'Enc_seq_pos: {enc_seq_pos.shape}')
        print(f'Dec_seq_pos: {dec_seq_pos.shape}')

        # Encoder Decoder
        enc_out = self.encoder(enc_seq_pos, enc_padding_mask)
        dec_out = self.decoder(dec_seq_pos, enc_out, look_ahead_mask, dec_padding_mask)
        print(f'Enc_out: {enc_out.shape}')
        print(f'Dec_out: {dec_out.shape}')

        # Output
        dec_out = self.dropout(dec_out)
        out = Dense(vocab_size, activation='softmax')(dec_out)
        print(f'Out: {out.shape}')

        return out


    def compute_mask(self, inp, tar):

        # for attention block of encoder
        enc_padding_mask = create_padding_mask(inp)

        # for 2nd attention head of decoder
        dec_padding_mask = create_padding_mask(inp)

        
        # For 1 attention head of decoder
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1].numpy())

        dec_target_padding_mask = create_padding_mask(tar)

        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)


        return enc_padding_mask, dec_padding_mask, look_ahead_mask



model = Transformer(embed_dim, laten_dim, num_heads, vocab_size, enc_seq_len, dec_seq_len)

temp_input = tf.random.uniform((64, enc_seq_len), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, dec_seq_len), dtype=tf.int64, minval=0, maxval=200)

out1 = model([temp_input, temp_target])

print(out1.shape)