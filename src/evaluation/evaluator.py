import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
)

from src.evaluation import metrics


class Evaluator:
    def __init__(
        self,
        features_path: str,
        cluster_labels_folder_path: str,
        image_names_path: str,
        ground_truth_path: str,
    ) -> None:
        """Inits an Evaluator instance.

        :param features_path: A string indicating the path of the features file.
        :param cluster_labels_folder_path: A string indicating the folder containing the
            cluster labels.
        :param image_names_path: A string indicating the path of the file containing the
            image_names for which the features were computed.
        :param ground_truth_path: A string indicating the file containing the ground
            truth.
        """
        self.features_path: str = features_path
        self.cluster_labels_folder_path: str = cluster_labels_folder_path
        self.image_names_path: str = image_names_path
        self.ground_truth_path: str = ground_truth_path
        self.scores: dict[str, float] = {}
        self.image_count: Optional[int] = None
        self.features: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.test_image_actual_labels: Optional[np.ndarray] = None
        self.test_image_cluster_labels: Optional[np.ndarray] = None

    def compute_metrics(self) -> None:
        """Computes the following scores:
        - silhouette score
        - davies-bouldin score
        - pairwise precision
        - pairwise recall
        - pairwise f1
        """
        self._load_data()

        self.scores["image_count"] = self.image_count
        self.scores["silhouette"] = silhouette_score(self.features, self.cluster_labels)
        self.scores["davies_bouldin"] = davies_bouldin_score(
            self.features, self.cluster_labels
        )
        self.scores["precision"] = metrics.pairwise_precision(
            self.test_image_actual_labels, self.test_image_cluster_labels
        )
        self.scores["recall"] = metrics.pairwise_recall(
            self.test_image_actual_labels, self.test_image_cluster_labels
        )
        self.scores["f1"] = metrics.pairwise_f1(
            self.test_image_actual_labels, self.test_image_cluster_labels
        )

    def save_metrics(self) -> None:
        """Saves the scores in JSON format to the self.cluster_labels_folder_path
        folder."""
        if len(self.scores) == 0:
            raise ValueError("Scores have not been computed.")

        with open(f"{self.cluster_labels_folder_path}/metrics.json", mode="w") as f:
            json.dump(self.scores, f)

    def _load_data(self) -> None:
        self.features = np.load(self.features_path)
        self.cluster_labels = np.load(
            f"{self.cluster_labels_folder_path}/cluster_labels.npy"
        )

        if os.path.exists(self.image_names_path):
            with open(self.image_names_path, mode="rb") as f:
                image_names = pickle.load(f)
        else:
            parent_path: str = Path(
                self.image_names_path
            ).parent.parent.parent.parent.absolute()
            with open(f"{parent_path}/image_names.pickle", mode="rb") as f:
                image_names = pickle.load(f)
        self.image_count = len(image_names)

        actual_labels_df: pd.DataFrame = pd.read_csv(
            self.ground_truth_path, usecols=["image_name", "integer_label"]
        )
        image_idx: dict[str, int] = {v: i for i, v in enumerate(image_names)}
        test_image_idx: list[int] = [
            image_idx[name]
            for name in actual_labels_df["image_name"]
            if name in image_idx
        ]

        self.test_image_cluster_labels: np.ndarray = self.cluster_labels[test_image_idx]
        self.test_image_actual_labels: np.ndarray = actual_labels_df[
            actual_labels_df["image_name"].isin(image_names)
        ]["integer_label"].to_numpy()
