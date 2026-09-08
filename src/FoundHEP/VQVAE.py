import tensorflow as tf
import numpy as np
import keras
import os
from .TransDer import TransEncoder, TransDecoder
from .vquant import VectorQuantizer, VectorQuantizerEMA
#os.environ["KERAS_BACKEND"] = "tensorflow"

class Sampling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1234)

    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = keras.ops.shape(z_mean)[0]
        dim = keras.ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch,dim), seed = self.seed_generator)

        return z_mean + keras.ops.exp(0.5 * z_log_var)* epsilon


class Encoder(keras.layers.Layer):
    def __init__(self, latent_dim = 32, hidden_dim = 64, name = "encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = keras.layers.Dense(hidden_dim, activation = "relu")
        self.dense_mean = keras.layers.Dense(latent_dim)
        self.dense_log_var = keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        return z_mean, z_log_var


class Decoder(keras.layers.Layer):
    def __init__(self, original_dim, hidden_dim = 64, name = "decoder", ** kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_proj = keras.layers.Dense(hidden_dim, activation = "relu")
        self.dense_output = keras.layers.Dense(original_dim, activation = "sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)









@keras.saving.register_keras_serializable()
class VQVAE(keras.Model):
    
    def __init__(self, original_dim, hidden_dim= 64, latent_dim = 32, num_heads = 16, train_variance=1.0, n_transf = 1, name = "vae", **kwargs):
        super().__init__(name = name, **kwargs)
        self.original_dim = int(original_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.train_variance = float(train_variance)
        self.n_transf = int(n_transf)



        self.encoder_input = keras.layers.Dense(self.hidden_dim, name = "encoder_input")
#        self.quantizer = VectorQuantizer(num_embeddings = 2048, embedding_dim = self.latent_dim, beta = 0.25)
        self.transencoder = []
        self.transdecoder = []
        for i in range(self.n_transf):
            self.transencoder.append(TransEncoder(out_dim = self.hidden_dim, num_heads = self.num_heads, name = "transEncoder_" +str(i)))

        self.to_latent = keras.layers.Dense(self.latent_dim)
        self.quantizer = VectorQuantizerEMA(num_embeddings = 512, embedding_dim = self.latent_dim, commitment_cost = 0.25, decay = 0.99)
        self.decoder_input = keras.layers.Dense(self.hidden_dim)

        for i in range(self.n_transf):
            self.transdecoder.append(TransEncoder(out_dim = self.hidden_dim, num_heads = self.num_heads, name = "transDecoder_" + str(i)))

        self.output_projection = keras.layers.Dense(self.original_dim)



        self.total_loss_tracker = keras.metrics.Mean(name = "total_loss")
        self.reco_loss_tracker = keras.metrics.Mean(name = "reco_loss")
        self.vq_loss_tracker = keras.metrics.Mean(name = "vq_loss")


    @property
    def metrics(self):
        return [
                self.total_loss_tracker,
                self.reco_loss_tracker,
                self.vq_loss_tracker
                ]

    def call(self, inputs, attention_mask = None, training = False):

        x = self.encoder_input(inputs)

        for enc in self.transencoder:
            x = enc(x, attention_mask = attention_mask, training=training)
        z_e = self.to_latent(x)
        vq_output = self.quantizer(z_e, training = training)
        embedded_input = vq_output["quantize"]
        dec_x = self.decoder_input(embedded_input)

        for dec in self.transdecoder:
            dec_x = dec(dec_x, attention_mask = attention_mask, training = training)
        #z_mean, z_log_var= self.encoder(x)
        #z = self.sampling((z_mean, z_log_var))
        #reco = self.decoder(z)

        #kl_loss = -0.5 * keras.ops.mean(z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var) + 1)
        #self.add_loss(kl_loss)
        return {
                "reco": self.output_projection(dec_x),
                "encoding_indices": vq_output["encoding_indices"]
                }
    



    def compute_loss(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
    ):
        reconstructions = y_pred["reco"]

        # Allows both:
        # model.fit(x)
        # model.fit(x, y)
        targets = x if y is None else y

        reconstruction_loss = keras.ops.mean(
            keras.ops.square(targets - reconstructions)
        )

        reconstruction_loss = reconstruction_loss / (
            self.train_variance + 1e-7
        )

        if self.losses:
            vq_loss = keras.ops.sum(
                keras.ops.stack(self.losses)
            )
        else:
            vq_loss = keras.ops.zeros(
                (),
                dtype=reconstructions.dtype,
            )

        total_loss = reconstruction_loss + vq_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reco_loss_tracker.update_state(
            reconstruction_loss
        )
        self.vq_loss_tracker.update_state(vq_loss)

        return total_loss



    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "original_dim": self.original_dim,
                    "hidden_dim": self.hidden_dim,
                    "latent_dim": self.latent_dim,
                    "num_heads": self.num_heads,
                    "train_variance": self.train_variance,
                    "n_transf": self.n_transf
                    })
        return config

    def compute_output_shape(self, input_shape):
        input_shape = tuple(input_shape)

        return {
            "reco": input_shape[:-1] + (
                self.original_dim,
            ),
            "encoding_indices": input_shape[:-1],
        }



