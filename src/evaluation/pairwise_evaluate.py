import pickle

import numpy as np
import pandas as pd

from src.evaluation import metrics


def pairwise_evaluate(
    features_dir: str, ground_truth_path: str
) -> tuple[float, float, float]:
    """Computes pairwise precision, recall, and f1 score based on the features found at
    the given folder using the ground truth file in the given path.

    :param features_dir: A string indicating the path of the features' directory.
    :param ground_truth_path: A string indicating the file containing the ground truth.
    :return: A 3-tuple containing the scores.
    """
    with open(f"{features_dir}/image_names.pickle", mode="rb") as f:
        image_names = pickle.load(f)
    cluster_labels: np.ndarray = np.load(f"{features_dir}/cluster_labels.npy")
    actual_labels_df: pd.DataFrame = pd.read_csv(
        ground_truth_path, usecols=["image_name", "integer_label"]
    )

    image_idx: dict[str, int] = {v: i for i, v in enumerate(image_names)}
    test_image_idx: list[int] = [
        image_idx[name] for name in actual_labels_df["image_name"] if name in image_idx
    ]
    test_image_cluster_labels: np.ndarray = cluster_labels[test_image_idx]

    test_image_actual_labels: np.ndarray = actual_labels_df[
        actual_labels_df["image_name"].isin(image_names)
    ]["integer_label"].to_numpy()

    precision: float = metrics.pairwise_precision(
        test_image_actual_labels, test_image_cluster_labels
    )
    recall: float = metrics.pairwise_recall(
        test_image_actual_labels, test_image_cluster_labels
    )
    f1: float = metrics.pairwise_f1(test_image_actual_labels, test_image_cluster_labels)
    return precision, recall, f1
