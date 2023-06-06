import numpy as np
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

from src.features.global_features.abstract_global_feature import (
    AbstractGlobalFeature,
)


class VGG16Feature(AbstractGlobalFeature):
    def __init__(
        self,
        resize_size: tuple[int, int] = None,
    ) -> None:
        """Inits a VGG16 instance.

        :param resize_size: A 2-tuple of integers indicating the pixel width and height
            of the resized image. This is useless for this feature.
        """
        super().__init__(resize_size)

        vgg16_model: VGG16 = VGG16()
        self.model: Model = Model(
            inputs=vgg16_model.inputs, outputs=vgg16_model.layers[-2].output
        )

    def read_image(self, image_path: str) -> np.ndarray:
        """Reads the image found in the given path as grayscale, resizes it based on the
        self.resize_size attribute and returns the image as a numpy array.

        :param image_path: A string indicating the path to the image.
        :return: A numpy array containing the image.
        """
        im: Image = Image.open(image_path).resize((224, 224))
        image: np.ndarray = np.array(im)
        return preprocess_input(np.resize(image, new_shape=(1, 224, 224, 3)))

    def compute_image_features(self, image: np.ndarray) -> np.ndarray:
        """Computes VGG16 features for the given image.

        :param image: A numpy array containing the image.
        :return: A numpy array containing the computed features.
        """
        return self.model.predict(image).ravel()
