from typing import Any
import numpy as np
import json


def combine_and_save(feature_paths: dict[str, str]) -> None:
    feature_files: dict[str, dict[str, Any]]
    for feature_name, feature_path in feature_paths.items():
        feature_files[feature_name] = _load_feature(feature_path)


def _load_feature(feature_path: str) -> dict[str, Any]:
    features: np.ndarray = np.load(f"{feature_path}/features.npy")
    with open(f"{feature_path}/feature_config.json", mode="r") as f:
        features_config: dict = json.load(f)

