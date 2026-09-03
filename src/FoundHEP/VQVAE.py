import tensorflow as tf
import numpy as np
import keras
import os
from .TransDer import TransEncoder, TransDecoder
#os.environ["KERAS_BACKEND"] = "tensorflow"

# Class from https://keras.io/examples/generative/vq_vae/ and https://arxiv.org/pdf/1711.00937

class VectorQuantizer(keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta ## The `beta` parameter is best kept between [0.25, 2] as per the paper.
        
        # Initialize the embeddings codebook
        self. embeddings = self.add_weight(shape = (self.embedding_dim, self.num_embeddings),
                                            initializer = "random_uniform",
                                            trainable = True,
                                            name = "embeddings_vqvae")
        


    def get_code_indices(self, flattened_inputs):
        # Calculate the L2-normalized distance
        similarity = keras.ops.matmul(flattened_inputs, self.embeddings)
        distances = (keras.ops.sum(keras.ops.square(flattened_inputs), axis = 1, keepdims = True) + keras.ops.sum(keras.ops.square(self.embeddings), axis = 0) - 2 * similarity)
        return keras.ops.argmin(distances, axis = 1)

    def call(self, inputs):
        input_shape = keras.ops.shape(inputs)
        flattened = keras.ops.reshape(inputs, [-1, self.embedding_dim])

        encoding_indices = self.get_code_indices(flattened)
        # Reshape indices to match spatial dimensions (e.g., 7x7)
        encoding_indices = keras.ops.reshape(encoding_indices, input_shape[:-1])

        encodings = keras.ops.one_hot(encoding_indices, self.num_embeddings)
        quantized = keras.ops.matmul(encodings, keras.ops.transpose(self.embeddings))
        quantized = keras.ops.reshape(quantized, input_shape)

        commitment_loss = keras.ops.mean((keras.ops.stop_gradient(quantized) - inputs) ** 2)
        codebook_loss = keras.ops.mean((quantized - keras.ops.stop_gradient(inputs)) **2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        quantized = x + keras.ops.stop_gradient(quantized - x)

        # RETURN BOTH: The quantized tensor and the indices
        return [quantized, encoding_indices]




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
        self.quantizer = VectorQuantizer(num_embeddings = 2048, embedding_dim = latent_dim)
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
    












