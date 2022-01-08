import tensorflow as tf
import numpy as np
import pandas as pd


inputs = tf.keras.Input(shape=(None,))

embedding = tf.keras.layers.Embedding(input_dim=10, output_dim=5)(inputs)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.units = units

        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True)

    def call(self, x):
        out, state_h, state_c = self.lstm(x)

        return out, state_h, state_c


class attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(attention, self).__init__()
        self.units = units

        self.W1 = tf.keras.layers.Dense(self.units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(self.units, use_bias=False)
        self.V = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, enc_out, decoder_hidden):

        decoder_hidden_reshaped = tf.expand_dims(decoder_hidden, axis=1)

        score = self.V(tf.nn.tanh(self.W1(enc_out) + self.W2(decoder_hidden_reshaped)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * enc_out

        return context_vector, attention_weights


class Decoder(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Decoder, self).__init__()
        self.units = units

        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(10)

        self.attention = attention(self.units)

    def call(self, x, encoder_hidden, encoder_cell, enc_outputs):

        out, dec_h, dec_c = self.lstm(x, initial_state=[encoder_hidden, encoder_cell])

        context_vector, attention_weights = self.attention(enc_outputs, dec_h)

        decoder_concat = tf.concat([context_vector, out], axis=-1)

        out = self.fc(decoder_concat)

        return out


encoder_out, hs, cs = Encoder(4)(embedding)

decoder_input = tf.keras.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(input_dim=10, output_dim=5)(decoder_input)
decoder_out = Decoder(4)(decoder_embedding, hs, cs, encoder_out)

model = tf.keras.Model(inputs=[inputs, decoder_input], outputs=[decoder_out])

random_input = np.array([1, 2, 3]).reshape(1, 3)
random_decoder_input = np.array([4, 5, 6]).reshape(1, 3)

out = model.predict([random_input, random_decoder_input])

print(out.shape)