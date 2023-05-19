from tensorflow.python.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers, layers, regularizers


class SparseAutoencoder(Model):
    def __init__(
        self, latent_dim: int, output_dim: int, lambda_: float, beta: float, p: float
    ) -> None:
        """Inits a SparseAutoencoder instance.

        :param latent_dim: An integer indicating the latent dimension of the encoder.
        :param output_dim: An integer indicating the output dimension of the decoder.
        :param lambda_: A float indicating the l2 regularization parameter.
        :param beta: A float indicating the sparsity regularization value.
        :param p: A float indicating the sparsity proportion value.
        """
        super().__init__(name="autoencoder")
        self.encoder: layers.Dense = layers.Dense(
            latent_dim,
            activation="sigmoid",
            kernel_initializer=initializers.initializers_v2.RandomNormal(
                mean=0.0, stddev=0.01
            ),
            bias_initializer=initializers.initializers_v2.RandomNormal(
                mean=0.0, stddev=0.01
            ),
            kernel_regularizer=regularizers.l2(lambda_ / 2),
            activity_regularizer=_get_sparse_regularizer(beta, p),
            name="encoder",
        )
        self.decoder: layers.Dense = layers.Dense(
            output_dim,
            activation="sigmoid",
            kernel_initializer=initializers.initializers_v2.RandomNormal(
                mean=0.0, stddev=0.01
            ),
            bias_initializer=initializers.initializers_v2.RandomNormal(
                mean=0.0, stddev=0.01
            ),
            kernel_regularizer=regularizers.l2(lambda_ / 2),
            name="decoder",
        )

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


def _get_sparse_regularizer(beta: float, p: float):
    def sparse_regularizer(activation_matrix):
        p_hat = K.mean(activation_matrix)
        kl_divergence = p * (K.log(p / p_hat)) + (1 - p) * (
            K.log((1 - p) / (1 - p_hat))
        )
        return beta * K.sum(kl_divergence)

    return sparse_regularizer
