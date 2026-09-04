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










class VQVAE(keras.Model):
    
    def __init__(self, original_dim, hidden_dim= 64, latent_dim = 32, name = "vae", **kwargs):
        super().__init__(name = name, **kwargs)
        data = np.load("/home/marco/FoundHEP/dataset.npy")
        
        print(data.shape)
        print(data[0])
        self.original_dim = original_dim
#        self.encoder = Encoder()
        self.quantizer = VectorQuantizerEMA(num_embeddings = 2048, embedding_dim = latent_dim, commitment_cost = 0.25, decay = 0.9)
#        self.decoder = Decoder(original_dim)
#        self.sampling = Sampling()
        self.transencoder = TransEncoder()
        self.transdecoder = TransDecoder()

    def call(self, inputs):

        encoded_input = self.transencoder(inputs)
        embedded_input = self.quantizer(encoded_input)
        decoded_latent = self.transdecoder(inputs, embedded_input)
        #z_mean, z_log_var= self.encoder(x)
        #z = self.sampling((z_mean, z_log_var))
        #reco = self.decoder(z)

        #kl_loss = -0.5 * keras.ops.mean(z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var) + 1)
        #self.add_loss(kl_loss)
        return decoded_latent
    












