from typing import cast

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Model

from src.dimensionality_reduction.abstract_reducer import AbstractReducer


class AutoencoderReducer(AbstractReducer):
    def __init__(self, autoencoder: Model, optimizer: str, loss: str) -> None:
        """Inits an AutoencoderReducer instance.

        :param autoencoder: An instance of keras.Model representing the autoencoder to
            use for dimensionality reduction.
        :param optimizer: A string indicating the optimizer to use.
        :param loss: A string indicating the loss function to use.
        """
        self.autoencoder: Model = autoencoder
        self.optimizer: str = optimizer
        self.loss: str = loss

        self.autoencoder.compile(optimizer=self.optimizer, loss=self.loss)

        self.min_max_scaler: MinMaxScaler = MinMaxScaler()

    def reduce_dimensions(
        self, samples: np.ndarray, epochs: int = 10, batch_size: int = 32
    ) -> np.ndarray:
        """Reduces the dimensions of the given samples.

        :param samples: A 2-d numpy array of shape (n_samples, n_features) containing
            the samples.
        :param epochs: An integer indicating the number of epochs.
        :param batch_size: A float indicating the batch size to use.
        :return: A 2-d numpy array of shape (n_samples, n_reduced_features) containing
            the samples in a latent space with a lower dimensionality.
        """
        train, test = train_test_split(samples, test_size=0.1)
        self.min_max_scaler = self.min_max_scaler.fit(train)
        self.min_max_scaler = cast(MinMaxScaler, self.min_max_scaler)

        scaled_train: np.ndarray = self.min_max_scaler.transform(train)
        scaled_test: np.ndarray = self.min_max_scaler.transform(test)

        self.autoencoder.fit(
            scaled_train,
            scaled_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(scaled_test, scaled_test),
        )

        return self.autoencoder.encoder(self.min_max_scaler.transform(samples)).numpy()
