import io
import os

import cv2 as cv
import numpy as np
from google.cloud import vision
from tqdm import tqdm


class TextCropper:
    @staticmethod
    def crop_text_and_save_panels(
        image_folder_path, block_confidence: float = 0.8
    ) -> None:
        """Crops text blocks from the images and overwrites them.

        :param image_folder_path: A string indicating the path to the folder containing
            the images.
        :param block_confidence: A float indicating the minimum confidence for the
            detected blocks. Defaults to 0.8.
        """
        for file in tqdm(
            os.listdir(image_folder_path),
            desc="Cropping Text Blocks - Completed Images",
        ):
            image_path = f"{image_folder_path}/{file}"
            bboxes = TextCropper.detect_text_blocks(image_path, block_confidence)
            cropped_image = TextCropper.crop_text_blocks(image_path, bboxes)
            cv.imwrite(image_path, cropped_image)

    @staticmethod
    def detect_text_blocks(
        path: str, block_confidence: float
    ) -> list[tuple[int, int, int, int]]:
        """Detect text blocks in the image found ion the given path. Detected text
        blocks have a confidence greater than the given block confidence.

        :param path: A string indicating the path to the image.
        :return: A list of 4-tuples (x_min, y_min, x_max, y_max) representing the
            bounding boxes of the detected blocks.
        :param block_confidence: A float indicating the minimum confidence for the
            detected blocks.
        """
        # Run detection
        client = vision.ImageAnnotatorClient()
        with io.open(path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(
                    response.error.message
                )
            )

        # Identify the blocks with high confidence
        blocks_vertices = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                if block.confidence < block_confidence:
                    continue
                blocks_vertices.append(block.bounding_box)

        # Convert the vertices to 4-tuples
        bboxes = []
        for bv in blocks_vertices:
            pts = np.array([[vertex.x, vertex.y] for vertex in bv.vertices], np.int32)
            xmin = np.min([x[0] for x in pts])
            ymin = np.min([x[1] for x in pts])
            xmax = np.max([x[0] for x in pts])
            ymax = np.max([x[1] for x in pts])
            bboxes.append((xmin, ymin, xmax, ymax))

        return bboxes

    @staticmethod
    def crop_text_blocks(
        path: str, bboxes: list[tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Crops the given text blocks from the image found in the given path.

        :param path: A string indicating the path to the image.
        :param bboxes: A list of 4-tuples (x_min, y_min, x_max, y_max) representing the
            bounding boxes of the detected blocks.
        :return: A numpy.ndarray representing the cropped images.
        """
        image = cv.imread(path)
        for b in bboxes:
            image[b[1] : b[3], b[0] : b[2]] = (255, 255, 255)
        return image
