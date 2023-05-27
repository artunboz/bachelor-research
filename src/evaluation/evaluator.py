import json
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.evaluation import metrics


class Evaluator:
    def __init__(self, features_dir: str, ground_truth_path: str) -> None:
        """Inits an Evaluator instance.

        :param features_dir: A string indicating the path of the directory containing
            the features.
        :param ground_truth_path: A string indicating the file containing the ground
            truth.
        """
        self.features_dir: str = features_dir
        self.ground_truth_path: str = ground_truth_path
        self.scores: dict[str, float] = {}
        self.features: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.test_image_actual_labels: Optional[np.ndarray] = None
        self.test_image_cluster_labels: Optional[np.ndarray] = None

    def compute_metrics(self) -> None:
        """Computes the following scores:
        - silhouette score
        - calinski-harabasz score
        - davies-bouldin score
        - pairwise precision
        - pairwise recall
        - pairwise f1
        """
        self._load_data()

        self.scores["silhouette"] = silhouette_score(self.features, self.cluster_labels)
        self.scores["calinski_harabasz"] = calinski_harabasz_score(
            self.features, self.cluster_labels
        )
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
        """Saves the scores in JSON format to the self.features_dir folder."""
        if len(self.scores) == 0:
            raise ValueError("Scores have not been computed.")

        with open(f"{self.features_dir}/scores/json", mode="w") as f:
            json.dump(self.scores, f)

    def _load_data(self) -> None:
        self.features = np.load(f"{self.features_dir}/features.npy")
        self.cluster_labels = np.load(f"{self.features_dir}/cluster_labels.npy")

        with open(f"{self.features_dir}/image_names.pickle", mode="rb") as f:
            image_names = pickle.load(f)
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
