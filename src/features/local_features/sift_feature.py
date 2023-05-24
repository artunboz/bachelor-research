import cv2 as cv
import numpy as np

from src.features.local_features.abstract_local_feature import AbstractLocalFeature


class SIFTFeature(AbstractLocalFeature):
    def __init__(
        self,
        resize_size: tuple[int, int],
        bovw_n_clusters_space: list[int],
        n_features: int = 10,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.09,
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
    ) -> None:
        """Inits a SIFTFeature instance. The underlying implementation relies on
        OpenCV's implementation of the SIFT feature. The parameter descriptions are
        taken from the documentation of OpenCV. For more details, please refer to
        https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html

        :param resize_size: A 2-tuple of integers indicating the pixel width and height
            of the resized image.
        :param bovw_n_clusters_space: A list of integers representing the search space
            for the optimal n_clusters value based on the silhouette score.
        :param n_features: An integer indicating the number of best features to retain.
        :param n_octave_layers: An integer indicating the number of layers in each
            octave.
        :param contrast_threshold: A float indicating the contrast threshold used to
            filter out weak features in semi-uniform (low-contrast) regions. The larger
            the threshold, the fewer features are produced by the detector.
        :param edge_threshold: A float indicating the threshold used to filter out
            edge-like features. The larger the edgeThreshold, the fewer features are
            filtered out (more features are retained).
        :param sigma: A float indicating the sigma of the Gaussian applied to the input
            image at the octave #0.
        """
        super().__init__(resize_size, bovw_n_clusters_space)
        self.n_features: int = n_features
        self.n_octave_layers: int = n_octave_layers
        self.contrast_threshold: float = contrast_threshold
        self.edge_threshold: float = edge_threshold
        self.sigma: float = sigma
        self.sift: cv.SIFT = cv.SIFT_create(
            nfeatures=self.n_features,
            nOctaveLayers=self.n_octave_layers,
            contrastThreshold=self.contrast_threshold,
            edgeThreshold=self.edge_threshold,
            sigma=self.sigma,
        )

    def read_image(self, image_path: str) -> np.ndarray:
        """Reads the image found in the given path as grayscale and resizes the image
        using the self.resize_size attribute.

        :param image_path: A string indicating the path to the image.
        :return: A numpy array containing the image.
        """
        image: np.ndarray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        return cv.resize(image, self.resize_size)

    def get_descriptors(self, image: np.ndarray) -> np.ndarray:
        """Computes the descriptors for the given image.

        :param image: A numpy array containing the image.
        :return: A numpy array containing the descriptors.
        """
        _, des = self.sift.detectAndCompute(image, None)
        return des
