import os
from abc import abstractmethod
from typing import Optional

import numpy as np
from fishervector import FisherVectorGMM
from tqdm import tqdm

from src.features.abstract_feature import AbstractFeature
from src.features.local_features.bag_of_visual_words import compute_bovw_features


class AbstractLocalFeature(AbstractFeature):
    def __init__(
        self,
        resize_size: tuple[int, int],
        quantization_method: str,
        n_components_space: list[int],
    ) -> None:
        """Inits and AbstractLocalFeature instance.

        :param resize_size: A 2-tuple of integers indicating the pixel width and height
            of the resized image.
        :param quantization_method: A sting indicating the quantization method for
            converting the local descriptors of an image to a single feature vector.
            Available options are "fisher" for fisher vectors and "bovw" for
            bag-of-visual-words.
        :param n_components_space: A list of integers containing either the number of
            components to use or multiple numbers that form the options to choose the
            best number from for vector quantization.
        """
        super().__init__(resize_size)
        self.quantization_method: str = quantization_method
        self.n_components_space: list[int] = n_components_space

    @abstractmethod
    def get_descriptors(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Computes the descriptors for the given image.

        :param image: A numpy array containing the image.
        :return: A numpy array containing the descriptors. If the computation was not
            successful for some reason, returns None.
        """

    def extract_features(self, image_folder_path: str) -> np.ndarray:
        """Extracts features from all the images found in the folder located at the
        given path and returns them in a 2-d numpy array of shape
        (n_images, n_features).

        :param image_folder_path: A string indicating the path to the folder containing
            the images.
        :return: A 2-d numpy array of shape (n_images, n_features).
        """
        sorted_image_names: list[str] = sorted(os.listdir(image_folder_path))
        image_name_descriptor_list: list[tuple[str, np.ndarray]] = []
        for image_name in tqdm(
            sorted_image_names,
            desc="Extracting features from the images",
        ):
            image: np.ndarray = self.read_image(f"{image_folder_path}/{image_name}")
            descriptors: np.ndarray = self.get_descriptors(image)
            if descriptors is not None:
                image_name_descriptor_list.append((image_name, descriptors))

        self.image_names, descriptor_list = list(zip(*image_name_descriptor_list))
        print("Quantizing vectors...")
        self.image_features = self._vector_quantization(descriptor_list)
        print("Features extracted!")

        return self.image_features

    def _vector_quantization(self, descriptor_list: list[np.ndarray]) -> np.ndarray:
        if self.quantization_method == "fisher":
            stacked_descriptors: np.ndarray = np.stack(descriptor_list)
            if len(self.n_components_space) > 1:
                fv_gmm: FisherVectorGMM = FisherVectorGMM().fit_by_bic(
                    stacked_descriptors, choices_n_kernels=self.n_components_space
                )
            else:
                fv_gmm: FisherVectorGMM = FisherVectorGMM(
                    n_kernels=self.n_components_space[0]
                ).fit(stacked_descriptors)
            fisher_vectors: np.ndarray = fv_gmm.predict(stacked_descriptors)
            return np.reshape(
                fisher_vectors,
                (
                    fisher_vectors.shape[0],
                    fisher_vectors.shape[1] * fisher_vectors.shape[2],
                ),
            )
        elif self.quantization_method == "bovw":
            return compute_bovw_features(descriptor_list, self.n_components_space)
        else:
            raise ValueError(
                f"The given quantization method of {self.quantization_method} is not"
                f" supported."
            )
