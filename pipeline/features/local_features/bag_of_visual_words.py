from typing import cast

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def compute_bovw_features(
    descriptor_list: list[np.ndarray], n_clusters_space: list[int]
) -> np.ndarray:
    """Computes bag-of-visual-words features for the given list of descriptors. The list
    contains 2-d numpy arrays of shape (n_keypoints, fixed_feature_length) that contain
    varying numbers of descriptors for certain images.

    :param descriptor_list: A list of 2-d numpy arrays of shape
        (n_keypoints, fixed_feature_length).
    :param n_clusters_space: A list of integers representing the search space for the
        optimal n_clusters value based on the silhouette score.
    """
    all_descriptors: np.ndarray = _get_stacked_descriptors(descriptor_list)
    optimal_kmeans: KMeans = _find_optimal_cluster_count(
        all_descriptors, n_clusters_space
    )
    bovw_features: np.ndarray = _extract_features(
        descriptor_list, optimal_kmeans.labels_, optimal_kmeans.n_clusters
    )
    return _l1_normalize(bovw_features)


def _get_stacked_descriptors(descriptor_list: list[np.ndarray]) -> np.ndarray:
    """Stacks the descriptors found in the given list of descriptors.

    :param descriptor_list: A list of 2-d numpy arrays of shape
        (n_keypoints, fixed_feature_length).
    :return: A 2-d numpy array of shape (n_descriptors, fixed_descriptor_length).
    """
    stacked_descriptors: np.ndarray = np.array(descriptor_list[0])
    for d in descriptor_list[1:]:
        stacked_descriptors = np.vstack((stacked_descriptors, d))

    return stacked_descriptors


def _find_optimal_cluster_count(
    descriptors: np.ndarray, n_clusters_space: list[int]
) -> KMeans:
    """Finds optimal n_cluster value from the given range based on the silhouette score
    of the resulting clusters.

    :param descriptors: A 2-d numpy array of shape
        (n_descriptors, fixed_descriptor_length).
    :param n_clusters_space: A list of integers representing the search space for the
        optimal n_clusters value based on the silhouette score.
    :return: An instance of sklearn.cluster.KMeans that is fitted on the given
        descriptors using the optimal n_clusters value.
    """
    kmeans_list: list[KMeans] = []
    scores: list[float] = []
    for n in tqdm(n_clusters_space, desc="Finding optimal n_clusters"):
        kmeans: KMeans = _cluster_descriptors(descriptors, n)
        score: float = silhouette_score(descriptors, kmeans.labels_)

        kmeans_list.append(kmeans)
        scores.append(score)

    optimal_idx: int = int(np.argmax(scores))
    return kmeans_list[optimal_idx]


def _cluster_descriptors(descriptors: np.ndarray, n_clusters: int) -> KMeans:
    """Clusters the given descriptors using sklearn.cluster.KMeans.

    :param descriptors: A 2-d numpy array of shape
        (n_descriptors, fixed_descriptor_length).
    :param n_clusters: An integer indicating the number of clusters.
    :return: A fitted instance of sklearn.cluster.KMeans.
    """
    return cast(KMeans, KMeans(n_clusters=n_clusters).fit(descriptors))


def _extract_features(
    descriptor_list: list[np.ndarray], labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Extracts the bag-of-visual-words histogram of each 2-d descriptor array in the
    given list.

    :param descriptor_list: A list of 2-d numpy arrays of shape
        (n_keypoints, fixed_feature_length).
    :param labels: A 1-d numpy array of shape containing the label for every descriptor
        found in the given list.
    :param n_clusters: An integer indicating the number of clusters.
    :return: A 2-d numpy array of shape (len(descriptor_list), n_clusters) containing
        the feature histograms.
    """
    im_features: np.ndarray = np.array(
        [np.zeros(n_clusters) for _ in range(len(descriptor_list))]
    )
    curr: int = 0
    for i, descriptors in enumerate(descriptor_list):
        n_labels: int = len(descriptors)
        descriptor_labels: np.ndarray = labels[curr : curr + n_labels]
        for label in descriptor_labels:
            im_features[i][label] += 1

        curr += n_labels

    return im_features


def _l1_normalize(histograms: np.ndarray) -> np.ndarray:
    """Normalizes the rows of the given 2-d numpy array using the L1 norm, i.e. the
    normalized rows sum to 1.

    :param histograms: A 2-d numpy array.
    :return: The l1-normalized 2-d numpy array.
    """
    rows_sums: np.ndarray = histograms.sum(axis=1, keepdims=True)
    return histograms / rows_sums
