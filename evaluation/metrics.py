from collections import defaultdict
from math import comb

import numpy as np


def pairwise_f1(actual_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    tp: int = _true_positive(actual_labels, cluster_labels)
    fp: int = _false_positive(actual_labels, cluster_labels)
    fn: int = _false_negative(actual_labels, cluster_labels)

    precision: float = tp / (tp + fp)
    recall: float = tp / (tp + fn)
    return (2 * precision * recall) / (precision + recall)


def pairwise_precision(actual_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    tp: int = _true_positive(actual_labels, cluster_labels)
    fp: int = _false_positive(actual_labels, cluster_labels)
    return tp / (tp + fp)


def pairwise_recall(actual_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    tp: int = _true_positive(actual_labels, cluster_labels)
    fn: int = _false_negative(actual_labels, cluster_labels)
    return tp / (tp + fn)


def _true_positive(actual_labels: np.ndarray, cluster_labels: np.ndarray) -> int:
    n_clusters: int = len(np.unique(cluster_labels))
    tp: int = 0
    for i in range(n_clusters):
        actual_cluster_labels: np.ndarray = actual_labels[cluster_labels == i]
        _, counts = np.unique(actual_cluster_labels, return_counts=True)
        for c in counts:
            tp += comb(c, 2)

    return tp


def _false_positive(actual_labels: np.ndarray, cluster_labels: np.ndarray) -> int:
    n_clusters: int = len(np.unique(cluster_labels))
    fp: int = 0
    for i in range(n_clusters):
        actual_cluster_labels: np.ndarray = actual_labels[cluster_labels == i]
        _, counts = np.unique(actual_cluster_labels, return_counts=True)
        for j, c in enumerate(counts):
            for k in range(j + 1, len(counts)):
                fp += c * counts[k]

    return fp


def _false_negative(actual_labels: np.ndarray, cluster_labels: np.ndarray) -> int:
    n_clusters: int = len(np.unique(cluster_labels))
    cluster_dist_list: list[dict[int, int]] = []
    for i in range(n_clusters):
        actual_cluster_labels: np.ndarray = actual_labels[cluster_labels == i]
        unique, counts = np.unique(actual_cluster_labels, return_counts=True)
        cluster_dist: dict[int, int] = {
            u: c for u, c in zip(list(unique), list(counts))
        }
        cluster_dist_list.append(cluster_dist)

    occurrence_counts: dict[int, list[int]] = defaultdict(list)
    for cd in cluster_dist_list:
        for k, v in cd.items():
            occurrence_counts[k].append(v)

    fn: int = 0
    for count_list in occurrence_counts.values():
        for i, c in enumerate(count_list):
            for j in range(i + 1, len(count_list)):
                fn += c * count_list[j]

    return fn
