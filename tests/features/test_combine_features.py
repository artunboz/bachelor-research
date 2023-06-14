import json
import pickle

import numpy as np

from paths import TEST_DATA_DIR
from src.features.combine_features import combine_and_save


def test_combine_features_regular():
    hog_path = f"{TEST_DATA_DIR}/combine_features/hog"
    lbp_path = f"{TEST_DATA_DIR}/combine_features/lbp"
    combined_path = f"{TEST_DATA_DIR}/combine_features/combined_regular"
    combine_and_save({"hog": hog_path, "lbp": lbp_path}, combined_path)

    expected_combined_features = np.array(
        [
            [0, 0, 0, 1, 1, 60, 1, 2, 4],
            [2, 2, 3, 4, 5, 70, 0, 0, 1],
        ]
    )
    actual_combined_features = np.load(f"{combined_path}/features.npy")
    np.testing.assert_array_equal(actual_combined_features, expected_combined_features)

    with open(f"{combined_path}/image_names.pickle", mode="rb") as f:
        actual_image_names = pickle.load(f)
    assert actual_image_names == ["1", "2"]

    expected_config = {
        "hog": {
            "resize_size": "64x64",
            "feature_dim": 5,
            "orientations": 6,
        },
        "lbp": {"resize_size": "48x48", "feature_dim": 4, "p": 24},
        "feature_dim": 9,
    }
    with open(f"{combined_path}/feature_config.json", mode="r") as f:
        actual_config = json.load(f)
    assert actual_config == expected_config


def test_combine_features_reduced():
    hog_path = f"{TEST_DATA_DIR}/combine_features/hog"
    lbp_reduced_path = f"{TEST_DATA_DIR}/combine_features/lbp/reductions/run_0"
    reduced_combined_path = f"{TEST_DATA_DIR}/combine_features/reduced_combined"
    combine_and_save(
        {"hog": hog_path, "lbp_reduced": lbp_reduced_path}, reduced_combined_path
    )

    expected_combined_features = np.array(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [2, 2, 3, 4, 5, 2, 2],
        ]
    )
    actual_combined_features = np.load(f"{reduced_combined_path}/features.npy")
    np.testing.assert_array_equal(actual_combined_features, expected_combined_features)

    with open(f"{reduced_combined_path}/image_names.pickle", mode="rb") as f:
        actual_image_names = pickle.load(f)
    assert actual_image_names == ["1", "2"]

    expected_config = {
        "hog": {
            "resize_size": "64x64",
            "feature_dim": 5,
            "orientations": 6,
        },
        "lbp_reduced": {
            "resize_size": "48x48",
            "feature_dim": 4,
            "p": 24,
            "reducer_config": {"n_components": 2},
        },
        "feature_dim": 7,
    }
    with open(f"{reduced_combined_path}/feature_config.json", mode="r") as f:
        actual_config = json.load(f)
    assert actual_config == expected_config
