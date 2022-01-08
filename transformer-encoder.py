import numpy as np
import tensorflow as tf
from tensorflow.keras import Input

# 1st step of creating a transformer encoder
# Create input embeddings for the transformer encoder

# let's say we are taking input like this.
encoder_input = Input(shape=(None,))

# Next step is to create the embedding layer
# we will use the embedding layer to create the embeddings

encoder_embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)(encoder_input)


# Now we will create a positional encoding layer
def get_angles(pos, i, d_dim):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_dim))
    return pos * angle_rates

pos_vector = tf.arange(0, seq_length)

pos_vector[:, 0::2] = np.sin(get_angles(pos_vector[:, 0::2], 0, d_dim))
pos_vector[:, 1::2] = np.cos(get_angles(pos_vector[:, 1::2], 1, d_dim))