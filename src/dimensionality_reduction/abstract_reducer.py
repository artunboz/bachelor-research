import json
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.util.helpers import create_json_dict


class AbstractReducer(ABC):
    def __init__(self) -> None:
        """Inits an AbstractReducer instance. Should not be used outside subclasses."""
        self.reduced_features: Optional[np.ndarray] = None

    @abstractmethod
    def reduce_dimensions(self, features_dir: str) -> np.ndarray:
        """Reduces the dimensions of the given samples.

        :param features_dir: A string indicating the file containing the features.
        :return: A 2-d numpy array of shape (n_samples, n_reduced_features) containing
            the samples in a latent space with a lower dimensionality.
        """

    def get_config(self) -> dict:
        """Returns the configuration of the reducer as a dictionary.

        :return: A dictionary containing the configuration of the reducer.
        """
        config: dict = create_json_dict(vars(self))
        return config

    def save_reduced_features(self, save_folder_path: str) -> None:
        """Saves the reduced features and the configuration of the reduction.

        :param save_folder_path: A sting indicating the folder to save the files to.
        """
        if self.reduced_features is None:
            raise ValueError("The features have not been reduced yet.")

        if not os.path.isdir(save_folder_path):
            os.makedirs(save_folder_path)

        with open(f"{save_folder_path}/reducer_config.json", mode="w") as f:
            json.dump(self.get_config(), f)
        np.save(f"{save_folder_path}/features.npy", self.reduced_features)
