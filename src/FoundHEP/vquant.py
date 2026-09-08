import math
import keras


# Class from https://keras.io/examples/generative/vq_vae/ and https://arxiv.org/pdf/1711.00937
@keras.saving.register_keras_serializable()
class VectorQuantizer(keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta ## The `beta` parameter is best kept between [0.25, 2] as per the paper.
        
        # Initialize the embeddings codebook
        self.embeddings = self.add_weight(shape = (self.embedding_dim, self.num_embeddings),
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

        quantized = inputs + keras.ops.stop_gradient(quantized - inputs)

        # RETURN BOTH: The quantized tensor and the indices
        return {
                "quantize" : quantized,
                "encoding_indices": encoding_indices,
                }

    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "num_embeddings": self.num_embeddings,
                    "embedding_dim": self.embedding_dim,
                    "beta": self.beta
                })
        return config

@keras.saving.register_keras_serializable()
class ExponentialMovingAverage(keras.layers.Layer):
    def __init__(self, decay, shape, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= decay <= 1:
            raise ValueError("Decay must be in range [0, 1]")
        self._decay = float(decay)
        self._shape = tuple(shape)
        self._counter = self.add_weight(shape = (), initializer = "zeros", trainable = False, dtype = "int64", name = "counter")

        self._hidden = self.add_weight(shape = self._shape, initializer = "zeros", trainable = False, name = "hidden")
        self._average = self.add_weight(shape = self._shape, initializer = "zeros", trainable = False, name = "average")


    def call(self, value):
        dtype = self._hidden.dtype
        value = keras.ops.cast(value, dtype)
        value = keras.ops.stop_gradient(value)

        decay = keras.ops.cast(self._decay, self._hidden.dtype)

        one = keras.ops.cast(1.0, dtype)

        self._counter.assign(self._counter + keras.ops.cast(1, "int64"))

        counter = keras.ops.cast(self._counter, dtype)
        
        new_hidden = (decay * self._hidden + (one - decay) * value)

        new_average = keras.ops.divide_no_nan(new_hidden, (one - keras.ops.power(decay, counter)))

        self._hidden.assign(new_hidden)
        self._average.assign(new_average)
        return new_average

    @property
    def value(self):
        return self._average

    def reset_state(self):
        self._hidden.assign(keras.ops.zeros_like(self._hidden))
        self._average.assign(keras.ops.zeros_like(self._average))
        self._counter.assign(keras.ops.zeros_like(self._counter))

    def get_config(self):
        config = super().get_config()
        config.update({
            "shape" : list(self._shape),
            "decay" : self._decay
            })
        return config



# Class from https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
@keras.saving.register_keras_serializable()
class VectorQuantizerEMA(keras.layers.Layer):
    def __init__(self,
                 embedding_dim,
                 num_embeddings,
                 commitment_cost,
                 decay,
                 epsilon = 1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive")

        if commitment_cost < 0.0:
            raise ValueError("commitment_cost must be non-negative")

        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in the range [0, 1)")

        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        self.embedding_dim = int(embedding_dim)
        self.num_embeddings = int(num_embeddings)
        self.decay = float(decay)
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)



        embedding_shape = (embedding_dim, num_embeddings)

        embedding_initializer = keras.initializers.VarianceScaling(
                scale = 1.0,
                mode = "fan_in",
                distribution = "uniform")
        # Initialize the embeddings codebook
        self.embeddings = self.add_weight(shape = embedding_shape,
                                            initializer = embedding_initializer,
                                            trainable = False,
                                            name = "embeddings_vqvae")

        # EMA of the number of assignments to every code.
        self.ema_cluster_size =  ExponentialMovingAverage(decay = self.decay, shape = (self.num_embeddings,), name = "ema_cluster_size")
                                
                                
                                
        # EMA of the sum of encoder vectors assigned to every code.
        self.ema_dw =  ExponentialMovingAverage(decay = self.decay, shape = embedding_shape,  name = "ema_wd")

    def call(self, inputs, training):
        inputs = keras.ops.cast(inputs, self.compute_dtype)
        input_shape = keras.ops.shape(inputs)
        flat_inputs = keras.ops.reshape(inputs, [-1, self.embedding_dim])
        
        encoding_indices = self.get_code_indices(flat_inputs)



        encodings = keras.ops.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = keras.ops.reshape(encoding_indices, input_shape[:-1])
        quantized = keras.ops.matmul(encodings, keras.ops.transpose(self.embeddings))
        quantized = keras.ops.reshape(quantized, input_shape)
        e_latent_loss = keras.ops.mean((keras.ops.stop_gradient(quantized) - inputs) ** 2)
        loss = keras.ops.cast(self.commitment_cost, e_latent_loss.dtype) * e_latent_loss
        self.add_loss(loss)


        if training:
            updated_ema_cluster_size = self.ema_cluster_size(keras.ops.sum(encodings, axis = 0))

            dw = keras.ops.einsum("nd,nk->dk", flat_inputs, encodings)
            updated_dw = self.ema_dw(dw)

            total_count = keras.ops.sum(updated_ema_cluster_size)

            smoothed_cluster_size = ((updated_ema_cluster_size + self.epsilon) / (total_count + self.num_embeddings * self.epsilon) * total_count)

            updated_embeddings = (updated_dw / keras.ops.expand_dims(smoothed_cluster_size, axis= 0))

            self.embeddings.assign(updated_embeddings)
        
        quantized = (inputs + keras.ops.stop_gradient(quantized - inputs))
        avg_probs = keras.ops.mean(encodings, axis = 0)
        perplexity = keras.ops.exp( -keras.ops.sum(avg_probs * keras.ops.log(avg_probs + keras.ops.cast(1e-10, avg_probs.dtype))))

        return {
                "quantize" : quantized,
                "perplexity": perplexity,
                "encodings": encodings,
                "encoding_indices": encoding_indices,
                }

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if (
            input_dim is not None
            and int(input_dim) != self.embedding_dim
        ):
            raise ValueError(
                "The final input dimension must equal embedding_dim. "
                f"Received input shape {input_shape} and "
                f"embedding_dim={self.embedding_dim}."
            )

        super().build(input_shape)


    def quantize(self, encoding_indices):
        """Looks up codebook vectors for arbitrary-shaped indices."""
        encoding_indices = keras.ops.cast(encoding_indices, "int32")

        # Convert codebook from:
        #     (embedding_dim, num_embeddings)
        # to:
        #     (num_embeddings, embedding_dim)
        codebook = keras.ops.transpose(self.embeddings)

        return keras.ops.take(
            codebook,
            encoding_indices,
            axis=0,
        )

    def get_code_indices(self, flattened_inputs):
        # Calculate the L2-normalized distance
        similarity = keras.ops.matmul(flattened_inputs, self.embeddings)
        distances = (keras.ops.sum(keras.ops.square(flattened_inputs), axis = 1, keepdims = True) + keras.ops.sum(keras.ops.square(self.embeddings), axis = 0) - 2 * similarity)
        return keras.ops.argmin(distances, axis = 1)
    
    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "embedding_dim": self.embedding_dim,
                    "num_embeddings": self.num_embeddings,
                    "commitment_cost": self.commitment_cost,
                    "decay": self.decay,
                    "epsilon": self.epsilon
                    })
        return config
    def compute_output_shape(self, input_shape):
        input_shape = tuple(input_shape)
        leading_shape = input_shape[:-1]

        if any(dim is None for dim in leading_shape):
            flat_size = None
        else:
            flat_size = math.prod(leading_shape)

        return {
            "quantize": input_shape,
            "perplexity": (),
            "encodings": (
                flat_size,
                self.num_embeddings,
            ),
            "encoding_indices": leading_shape,
        }
