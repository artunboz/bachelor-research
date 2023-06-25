import json
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.util.helpers import create_json_dict


class AbstractClustering(ABC):
    def __init__(self) -> None:
        """Inits an AbstractClustering instance. Should not be used outside subclasses."""
        self.cluster_labels: Optional[np.ndarray] = None

    @abstractmethod
    def cluster(self, features_dir: str) -> np.ndarray:
        """Clusters the given samples.

        :param features_dir: A string indicating the file containing the features.
        :return: A 1-d numpy array of shape containing the cluster label for each sample
            in the same order as the input array.
        """

    def get_config(self) -> dict:
        """Returns the configuration of the clustering as a dictionary.

        :return: A dictionary containing the configuration of the clustering.
        """
        config: dict = create_json_dict(vars(self))
        return config

    def save_cluster_labels(self, save_folder_path: str) -> None:
        """Saves the cluster labels and the configuration of the clustering.

        :param save_folder_path: A sting indicating the folder to save the files to.
        """
        if self.cluster_labels is None:
            raise ValueError("The clustering has not been done yet.")

        if not os.path.isdir(save_folder_path):
            os.makedirs(save_folder_path)

        with open(f"{save_folder_path}/clustering_config.json", mode="w") as f:
            json.dump(self.get_config(), f)
        np.save(f"{save_folder_path}/cluster_labels.npy", self.cluster_labels)
