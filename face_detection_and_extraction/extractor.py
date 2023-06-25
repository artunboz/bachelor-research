import os
import warnings

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def extract_images(
    image_folder_path,
    boxes_filename="boxes.csv",
    faces_folder_name="face_images",
    bodies_folder_name="body_images",
    visualize=False,
):
    faces_path = f"{image_folder_path}/{faces_folder_name}"
    bodies_path = f"{image_folder_path}/{bodies_folder_name}"
    if not os.path.exists(faces_path):
        os.makedirs(faces_path)
    if not os.path.exists(bodies_path):
        os.makedirs(bodies_path)

    boxes_df = pd.read_csv(f"{image_folder_path}/{boxes_filename}")
    index_image_mapping = {}
    for index, row in tqdm(boxes_df.iterrows(), desc="Boxes", total=boxes_df.shape[0]):
        img = cv2.imread(f"{image_folder_path}/{row['image_name']}")
        extracted_img = img[row["y0"] : row["y1"], row["x0"] : row["x1"]]
        if row["type"] == 0:
            path = f"{faces_path}/{index}.png"
        elif row["type"] == 1:
            path = f"{bodies_path}/{index}.png"
        else:
            warnings.warn(f"Unrecognized box type at {row['image_name']}")
            continue
        cv2.imwrite(path, extracted_img)
        index_image_mapping[index] = row["image_name"]

        if visualize:
            plt.imshow(extracted_img[:, :, ::-1])
            plt.show()
