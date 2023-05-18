import numpy as np
import pytest

from pipeline.features.local_features import bag_of_visual_words


@pytest.fixture
def example_descriptor_list():
    descriptors_1 = np.array([[1, 2], [2, 2]])
    descriptors_2 = np.array([[2, 3], [8, 7], [8, 9]])
    descriptors_3 = np.array([[8, 8], [25, 10]])

    return [descriptors_1, descriptors_2, descriptors_3]


@pytest.fixture
def example_descriptors():
    return np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 9], [8, 8], [25, 10]])


def test__get_stacked_descriptors(example_descriptor_list):
    actual_stacked_descriptors = bag_of_visual_words._get_stacked_descriptors(
        example_descriptor_list
    )
    expected_stacked_descriptors = np.array(
        [[1, 2], [2, 2], [2, 3], [8, 7], [8, 9], [8, 8], [25, 10]]
    )

    np.testing.assert_array_equal(
        actual_stacked_descriptors, expected_stacked_descriptors
    )


def test__find_optimal_cluster_count(example_descriptors):
    optimal_kmeans = bag_of_visual_words._find_optimal_cluster_count(
        example_descriptors, n_clusters_range=(2, 7)
    )

    assert optimal_kmeans.n_clusters == 3


def test__extract_features(example_descriptors, example_descriptor_list):
    actual_features = bag_of_visual_words._extract_features(
        example_descriptor_list, [0, 0, 0, 2, 2, 2, 1], 3
    )
    expected_features = np.array([[2, 0, 0], [1, 0, 2], [0, 1, 1]])

    np.testing.assert_array_equal(actual_features, expected_features)


def test__l1_normalize():
    test_matrix = np.array([[1, 3, 4], [0, 2, 0], [1, 1, 1]])
    expected_normalized_matrix = np.array(
        [[0.125, 0.375, 0.5], [0, 1, 0], [1 / 3, 1 / 3, 1 / 3]]
    )
    actual_normalized_matrix = bag_of_visual_words._l1_normalize(test_matrix)

    np.testing.assert_array_equal(actual_normalized_matrix, expected_normalized_matrix)
