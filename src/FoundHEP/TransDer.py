import keras
from .custom_layers import HeadScaledMultiHeadAttention


@keras.saving.register_keras_serializable()
class TransEncoder(keras.layers.Layer):
    def __init__(self, ctxt_dim = 0, num_heads = 16, out_dim = 32, dropout = 0.0, dense_nodes = 128, **kwargs):
        super().__init__(**kwargs)
        self.ctxt_dim = int(ctxt_dim)
        self.num_heads = int(num_heads)
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)
        self.dense_nodes = int(dense_nodes)
        if out_dim % num_heads != 0:
            raise ValueError(
                "out_dim must be divisible by num_heads"
            )
        key_dim = self.out_dim // self.num_heads
        self.atn_self = HeadScaledMultiHeadAttention(num_heads = self.num_heads, output_shape = self.out_dim, key_dim = key_dim, dropout = self.dropout)
        self.ff_step = keras.layers.Dense(self.dense_nodes, activation = "silu")
        self.transout = keras.layers.Dense(self.out_dim)

        self.norm_layer_atn_1 = keras.layers.LayerNormalization()
        self.norm_layer_atn_2 = keras.layers.LayerNormalization()

        self.norm_layer_ff_1 = keras.layers.LayerNormalization()
        self.norm_layer_ff_2 = keras.layers.LayerNormalization()

        self.dropout_1 = keras.layers.Dropout(self.dropout)
        self.dropout_2 = keras.layers.Dropout(self.dropout)

    def call(self, inputs, attention_mask = None, training = False):
        norm_inps = self.norm_layer_atn_1(inputs)
        atn = self.atn_self(query = norm_inps,
                            key = norm_inps,
                            value = norm_inps,
                            attention_mask = attention_mask,
                            training = training)
        atn = self.norm_layer_atn_2(atn)
        atn = self.dropout_1(atn, training = training)
        x = inputs + atn
        ff = self.norm_layer_ff_1(x)
        ff = self.ff_step(ff)
        ff = self.norm_layer_ff_2(ff)
        ff = self.transout(ff)
        ff = self.dropout_2(ff, training = training)

        return x + ff
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "ctxt_dim": self.ctxt_dim,
            "num_heads": self.num_heads,
            "out_dim": self.out_dim,
            "dropout": self.dropout,
            "dense_nodes": self.dense_nodes
            })
        return config


@keras.saving.register_keras_serializable()
class TransDecoder(keras.layers.Layer):
    def __init__(self, ctxt_dim = 0, num_heads = 16, out_dim = 32, dropout = 0.0, dense_nodes = 128, **kwargs):
        super().__init__(**kwargs)
        self.ctxt_dim = int(ctxt_dim)
        self.num_heads = int(num_heads)
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)
        self.dense_nodes = int(dense_nodes)
        if out_dim % num_heads != 0:
            raise ValueError(
                "out_dim must be divisible by num_heads"
            )
        key_dim = self.out_dim // self.num_heads
        self.atn_self = HeadScaledMultiHeadAttention(num_heads = self.num_heads, output_shape = self.out_dim, key_dim = key_dim, dropout = self.dropout)
        self.atn_cross = HeadScaledMultiHeadAttention(num_heads = self.num_heads, output_shape = self.out_dim, key_dim = key_dim, dropout = self.dropout)
        self.ff_step = keras.layers.Dense(self.dense_nodes, activation = "silu")
        self.transout = keras.layers.Dense(self.out_dim)

        self.norm_layer_atn_self_1 = keras.layers.LayerNormalization()
        self.norm_layer_atn_self_2 = keras.layers.LayerNormalization()

        self.norm_layer_atn_cross_1 = keras.layers.LayerNormalization()
        self.norm_layer_atn_cross_2 = keras.layers.LayerNormalization()

        self.norm_layer_ff_1 = keras.layers.LayerNormalization()
        self.norm_layer_ff_2 = keras.layers.LayerNormalization()

        self.dropout_self = keras.layers.Dropout(self.dropout)
        self.dropout_cross = keras.layers.Dropout(self.dropout)
        self.dropout_2 = keras.layers.Dropout(self.dropout)

    def call(self, inputs, enc_output, self_attention_mask = None, cross_attention_mask = None, training = False):

        norm_inps = self.norm_layer_atn_self_1(inputs)
        atn = self.atn_self(query = norm_inps,
                            key = norm_inps,
                            value = norm_inps,
                            attention_mask = self_attention_mask,
                            use_causal_mask = True,
                            training = training)
        atn = self.norm_layer_atn_self_2(atn)
        atn = self.dropout_self(atn, training = training)
        x = inputs + atn

        atn = self.norm_layer_atn_cross_1(x)
        atn = self.atn_cross(query = atn,
                            key = enc_output,
                            value = enc_output,
                            attention_mask = cross_attention_mask,
                            use_causal_mask = False,
                            training = training)
        atn = self.norm_layer_atn_cross_2(atn)
        atn = self.dropout_cross(atn, training = training)
        x = x + atn

        ff = self.norm_layer_ff_1(x)
        ff = self.ff_step(ff)
        ff = self.norm_layer_ff_2(ff)
        ff = self.transout(ff)
        ff = self.dropout_2(ff, training = training)

        return x + ff
    def get_config(self):
        config = super().get_config()
        config.update({
            "ctxt_dim": self.ctxt_dim,
            "num_heads": self.num_heads,
            "out_dim": self.out_dim,
            "dropout": self.dropout,
            "dense_nodes": self.dense_nodes
            })
        return config
