from tensorflow.python.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers, regularizers


class SparseAutoencoder(Model):
    def __init__(self, latent_dim: int, output_dim: int, lambda_: float) -> None:
        """Inits a SparseAutoencoder instance.

        :param latent_dim: An integer indicating the latent dimension of the encoder.
        :param output_dim: An integer indicating the output dimension of the decoder.
        :param lambda_: A float indicating the l2 regularization parameter.
        """
        super().__init__(name="autoencoder")
        self.encoder: layers.Dense = layers.Dense(
            latent_dim,
            activation="sigmoid",
            kernel_regularizer=regularizers.l2(lambda_ / 2),
            activity_regularizer=_sparse_regularizer,
            name="encoder",
        )
        self.decoder: layers.Dense = layers.Dense(
            output_dim,
            activation="sigmoid",
            kernel_regularizer=regularizers.l2(lambda_ / 2),
            name="decoder",
        )

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


def _sparse_regularizer(activation_matrix):
    """Taken from https://github.com/sudharsan13296/Hands-On-Deep-Learning-Algorithms-with-Python/blob/master/10.%20Reconsturcting%20Inputs%20using%20Autoencoders/10.09%20Building%20the%20Sparse%20Autoencoder.ipynb
    Credits to @sudharsan13296
    """
    p = 0.05
    beta = 3
    p_hat = K.mean(activation_matrix)

    KL_divergence = p * (K.log(p / p_hat)) + (1 - p) * (K.log(1 - p / 1 - p_hat))

    sum = K.sum(KL_divergence)

    return beta * sum
