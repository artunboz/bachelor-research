import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def combine_and_save(
    feature_folder_paths: dict[str, str], save_folder_path: str
) -> None:
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    feature_files: dict[str, dict[str, Any]] = {}
    for feature_name, feature_folder_path in feature_folder_paths.items():
        feature_files[feature_name] = _load_feature(feature_folder_path)

    combined_features: dict[str, Any] = _combine_features(feature_files)
    np.save(f"{save_folder_path}/features.npy", combined_features["features"])
    with open(f"{save_folder_path}/feature_config.json", mode="w") as f:
        json.dump(combined_features["feature_config"], f)
    with open(f"{save_folder_path}/image_names.pickle", mode="wb") as f:
        pickle.dump(combined_features["image_names"], f)


def _load_feature(feature_path: str) -> dict[str, Any]:
    if os.path.exists(f"{feature_path}/image_names.pickle"):
        return _load_regular_feature(feature_path)
    else:
        return _load_reduced_feature(feature_path)


def _load_regular_feature(feature_path: str) -> dict[str, Any]:
    features: np.ndarray = np.load(f"{feature_path}/features.npy")
    with open(f"{feature_path}/feature_config.json", mode="r") as f:
        feature_config: dict = json.load(f)
    with open(f"{feature_path}/image_names.pickle", mode="rb") as f:
        image_names: list[str] = pickle.load(f)

    return {
        "features": features,
        "feature_config": feature_config,
        "image_names": image_names,
    }


def _load_reduced_feature(feature_path: str) -> dict[str, Any]:
    parent_path: str = Path(feature_path).parent.absolute()
    features: np.ndarray = np.load(f"{feature_path}/features.npy")
    with open(f"{parent_path}/feature_config.json", mode="r") as f:
        feature_config: dict = json.load(f)
    with open(f"{feature_path}/reducer_config.json", mode="r") as f:
        reducer_config: dict = json.load(f)
    feature_config["reducer_config"] = reducer_config
    with open(f"{parent_path}/image_names.pickle", mode="rb") as f:
        image_names: list[str] = pickle.load(f)

    return {
        "features": features,
        "feature_config": feature_config,
        "image_names": image_names,
    }


def _combine_features(feature_files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # Combine the config dictionaries to a single dictionary.
    combined_config_dict: dict[str, dict] = {}
    for name, files in feature_files.items():
        combined_config_dict[name] = files["feature_config"]
    combined_config_dict["feature_dim"] = sum(
        [v["feature_dim"] for v in combined_config_dict.values()]
    )

    # Get the images for which all feature methods successfully computed features.
    image_names_list: list[list[str]] = [
        v["image_names"] for v in feature_files.values()
    ]
    common_image_names: set[str] = set(image_names_list[0])
    for image_names in image_names_list[1:]:
        common_image_names.intersection_update(image_names)
    new_image_names_list: list[str] = sorted(list(common_image_names))

    # Get the actual features for these common images.
    common_features_list = []
    for name, files in feature_files.items():
        image_names: list[str] = files["image_names"]
        image_names_idx: dict = dict((n, i) for i, n in enumerate(image_names))
        indices: list[int] = [image_names_idx[x] for x in new_image_names_list]
        features: np.ndarray = files["features"]
        common_features: np.ndarray = features[indices]
        common_features_list.append(common_features)
    stacked_common_features: np.ndarray = np.concatenate(common_features_list, axis=1)

    return {
        "features": stacked_common_features,
        "feature_config": combined_config_dict,
        "image_names": new_image_names_list,
    }
