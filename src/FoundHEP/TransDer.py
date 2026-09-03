import numpy as np
import tensorflow as tf
import keras


class TransEncoder(keras.layers.Layer):
    def __init__(self, ctxt_dim = 0, num_heads = 16, out_dim = 32, dropout = 0.0, dense_nodes = 128, **kwargs):
        super().__init__(**kwargs)
        key_dim = out_dim // num_heads
        self.ctxt_dim = ctxt_dim
        self.atn_head = keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = key_dim, dropout = dropout)
        self.ff_step = keras.layers.Dense(dense_nodes, activation = "silu")
        self.transout = keras.layers.Dense(out_dim)

        self.norm_layer_atn_1 = keras.layers.LayerNormalization()
        self.norm_layer_atn_2 = keras.layers.LayerNormalization()

        self.norm_layer_ff_1 = keras.layers.LayerNormalization()
        self.norm_layer_ff_2 = keras.layers.LayerNormalization()

        #self.dropout_1 = keras.layers.Dropout(dropout)
        self.dropout_2 = keras.layers.Dropout(dropout)

    def call(self, inputs, attention_mask = None, training = False):
        norm_inps = self.norm_layer_atn_1(inputs)
        atn = self.atn_head(query = norm_inps,
                            key = norm_inps,
                            value = norm_inps,
                            attention_mask = attention_mask,
                            training = training)
        norm_atn = self.norm_layer_atn_2(atn)
        x = inputs + norm_atn
        #x = self.dropout_1(x)
        ff = self.norm_layer_ff_1(x)
        ff = self.ff_step(ff)
        ff = self.norm_layer_ff_2(ff)
        ff = self.dropout_2(ff)
        ff = self.transout(ff)

        return x + ff
