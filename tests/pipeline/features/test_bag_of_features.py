import numpy as np
import pytest

from pipeline.features.bag_of_features import get_bof_features


@pytest.fixture
def example_descriptors():
    descriptors_1 = np.array([[1, 2], [2, 2]])
    descriptors_2 = np.array([[2, 3], [8, 7], [8, 9]])
    descriptors_3 = np.array([[8, 8], [25, 10]])

    return {"1": descriptors_1, "2": descriptors_2, "3": descriptors_3}


def test_no_fuzzy(example_descriptors):
    # Test case for when there are no fuzzy labels.
    # Labels for the descriptors in example_descriptors are
    # [0 0 0 1 1 1 2]
    expected_features = {
        "1": np.array([1, 0, 0]),
        "2": np.array([0.4472136, 0.89442719, 0]),
        "3": np.array([0, 0.70710678, 0.70710678]),
    }
    actual_features, n_clusters = get_bof_features(example_descriptors, 3, 1, "L2")

    assert actual_features.keys() == expected_features.keys()
    assert n_clusters == 3
    np.testing.assert_almost_equal(
        list(actual_features.values()), list(expected_features.values())
    )


def test_single_fuzzy(example_descriptors):
    # Test case for when there is a fuzzy label but the image containing the fuzzy
    # descriptor also has non-fuzzy labels.
    # Labels for the descriptors in example_descriptors are
    # [ 0  0  0  1  1  1 -1  0  2  2]
    example_descriptors["4"] = np.array([[1, 2], [100, 100], [101, 101]])
    expected_features = {
        "1": np.array([1, 0, 0]),
        "2": np.array([0.4472136, 0.89442719, 0]),
        "3": np.array([0, 1, 0]),
        "4": np.array([0.4472136, 0, 0.89442719]),
    }

    actual_features, n_clusters = get_bof_features(example_descriptors, 3, 2, "L2")

    assert actual_features.keys() == expected_features.keys()
    assert n_clusters == 3
    np.testing.assert_almost_equal(
        list(actual_features.values()), list(expected_features.values())
    )


def test_all_fuzzy(example_descriptors):
    # Test case for when an image only contains fuzzy labels.
    # Labels for the descriptors in example_descriptors are
    # [ 0  0  0  1  1  1 -1 -1 -1]
    example_descriptors["4"] = np.array([[100, 100], [200, 200]])
    expected_features = {
        "1": np.array([1, 0]),
        "2": np.array([0.4472136, 0.89442719]),
        "3": np.array([0, 1]),
    }

    actual_features, n_clusters = get_bof_features(example_descriptors, 3, 2, "L2")

    assert actual_features.keys() == expected_features.keys()
    assert n_clusters == 2
    np.testing.assert_almost_equal(
        list(actual_features.values()), list(expected_features.values())
    )
