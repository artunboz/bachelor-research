import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

from pipeline.features.abstract_feature import AbstractFeature
from pipeline.features.bag_of_features import get_bof_features


class ORBFeature(AbstractFeature):
    def __init__(
        self, resize_size: tuple[int, int], n_keypoints: int, patch_size: int
    ) -> None:
        """Inits a ORBFeature instance. The underlying implementation relies on OpenCV's
        implementation of the ORB feature.

        :param resize_size: A 2-tuple of integers indicating the pixel width and height
            of the resized image.
        :param n_keypoints: An integer indicating the maximum number of keypoints to
            detect in an image.
        :param patch_size: An integer indicating the value to use for the edgeThreshold
            and patchSize parameters of OpenCV's ORB implementation.
        """
        self.resize_size: tuple[int, int] = resize_size
        self.orb: cv.ORB = cv.ORB_create(
            nfeatures=n_keypoints, edgeThreshold=patch_size, patchSize=patch_size
        )

    def compute_features(
        self,
        image_folder_path: str,
        eps: float = 100.0,
        min_samples: int = 20,
        norm: str = "L2",
    ) -> np.ndarray:
        """Computes ORB features for all the images found in the folder located at the
        given path and return them in a 2-d numpy array of shape (n_images, n_features).

        :param image_folder_path: A string indicating the path to the folder containing
            the images.
        :param eps: A float indicating the eps parameter for DBSCAN.
        :param min_samples: An integer indicating the min_samples parameter for DBSCAN.
        :param norm: A string indicating the normalization technique to use when
            normalizing bag-of-feature histograms to get the final feature vectors.
        :return: A 2-d numpy array of shape (n_images, n_features).
        """
        sorted_image_names: list[str] = sorted(os.listdir(image_folder_path))
        descriptors_dict: dict[str, np.ndarray] = {}
        for image_name in tqdm(
            sorted_image_names,
            desc="ORB features extracted from images",
        ):
            image: np.ndarray = cv.imread(
                f"{image_folder_path}/{image_name}", cv.IMREAD_GRAYSCALE
            )
            descriptors: np.ndarray = self._get_orb_descriptors(image)
            if descriptors is None:
                continue

            descriptors_dict[image_name] = descriptors

        features_dict, feature_dim = get_bof_features(
            descriptors_dict, eps, min_samples, norm
        )
        features: np.ndarray = np.empty(
            shape=(len(os.listdir(image_folder_path)), feature_dim)
        )
        for i, image_name in enumerate(sorted_image_names):
            if image_name in features_dict:
                features[i] = features_dict[image_name]
            else:
                features[i] = np.zeros(feature_dim)

        return features

    def _get_orb_descriptors(self, image: np.ndarray) -> np.ndarray:
        """The given image must be grayscale."""
        resized_image = cv.resize(image, self.resize_size)
        _, des = self.orb.detectAndCompute(resized_image, None)
        return des
