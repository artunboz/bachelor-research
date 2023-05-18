import numpy as np
from skimage import feature
from skimage import io
from skimage import transform

from pipeline.features.global_features.abstract_global_feature import (
    AbstractGlobalFeature,
)


class LBPFeature(AbstractGlobalFeature):
    def __init__(
        self,
        resize_size: tuple[int, int],
        p: int = 8,
        r: int = 1,
        method: str = "uniform",
    ) -> None:
        super().__init__(resize_size)
        self.p: int = p
        self.r: int = r
        self.method: str = method

    def read_image(self, image_path: str) -> np.ndarray:
        image: np.ndarray = io.imread(image_path, as_gray=True)
        return transform.resize(image, self.resize_size[::-1])

    def compute_image_features(self, image: np.ndarray) -> np.ndarray:
        return feature.local_binary_pattern(image, self.p, self.r, self.method).ravel()
