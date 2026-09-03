import keras

@keras.saving.register_keras_serializable()
class HeadScaledMultiHeadAttention(keras.layers.MultiHeadAttention):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(
            num_heads=num_heads,
            key_dim=key_dim,
            **kwargs,
        )

        # One learned scalar per head, initialized as identity.
        self.head_scale = self.add_weight(
            name="head_scale",
            shape=(num_heads,),
            initializer="ones",
            trainable=True,
        )

    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        training=None,
        return_attention_scores=False,
        **kwargs,
    ):
        # attention_output: (batch, target_length, num_heads, value_dim)
        attention_output, attention_scores = super()._compute_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            training=training,
            return_attention_scores=return_attention_scores,
            **kwargs,
        )

        # (num_heads,) -> (1, 1, num_heads, 1)
        scale = keras.ops.reshape(
            keras.ops.cast(self.head_scale, attention_output.dtype),
            (1, 1, -1, 1),
        )

        attention_output = attention_output * scale

        return attention_output, attention_scores

