import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import losses
from tensorflow.python.keras.layers import Dense

from pipeline.dimensionality_reduction.abstract_reducer import AbstractReducer


class AutoencoderReducer(AbstractReducer):
    def __init__(
        self, actual_dim: int, latent_dim: int, hidden_layers: list[int]
    ) -> None:
        """Inits an AutoencoderReducer instance.

        :param actual_dim: An integer indicating the number of dimensions of original
            data.
        :param latent_dim: An integer indicating the number of dimensions to encode to.
        :param hidden_layers: A list of integers indicating the number of neurons in the
            layers up to the final encoder layer.
        """
        self.actual_dim: int = actual_dim
        self.latent_dim: int = latent_dim
        self.hidden_layers: list[int] = hidden_layers
        layers: list[int] = [*self.hidden_layers, self.latent_dim]
        self.autoencoder: Autoencoder = Autoencoder(
            layers=layers, output_units=self.actual_dim
        )
        self.autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())
        self.standard_scaler: MinMaxScaler = MinMaxScaler()

    def reduce_dimensions(self, samples: np.ndarray) -> np.ndarray:
        """Reduces the dimensions of the given samples.

        :param samples: A 2-d numpy array of shape (n_samples, n_features) containing
            the samples.
        :return: A 2-d numpy array of shape (n_samples, n_reduced_features) containing
            the samples in a latent space with a lower dimensionality.
        """
        # Minmax scaling on each row
        scaled_samples: np.ndarray = self.standard_scaler.fit_transform(samples.T).T
        train, test = train_test_split(scaled_samples, test_size=0.1)
        self.autoencoder.fit(
            train, train, epochs=15, batch_size=32, validation_data=(test, test)
        )

        return self.autoencoder.encoder.predict(scaled_samples, batch_size=32)


class Autoencoder(Model):
    def __init__(self, layers: list[int], output_units: int):
        """Inits an Autoencoder.

        :param layers: A list of integers indicating the number of neurons in the layers
            of the encoder.
        :param output_units: An integer indicating the number of output units for
            decoder.
        """
        super().__init__()

        encoder_layers: list[Dense] = [
            Dense(layer_size, activation="relu") for layer_size in layers
        ]
        decoder_layers: list[Dense] = [
            Dense(layer_size, activation="relu") for layer_size in layers[-2::-1]
        ]
        decoder_layers.append(Dense(output_units, activation="sigmoid"))

        self.encoder: Sequential = Sequential(encoder_layers)
        self.decoder: Sequential = Sequential(decoder_layers)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
