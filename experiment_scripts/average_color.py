import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

from paths import DATA_DIR

image_folder_path = f"{DATA_DIR}/cleaned_panels/mk_2/face_images"

image_names = os.listdir(image_folder_path)
n_images = len(image_names)
mean_arrs = {
    "b": [],
    "g": [],
    "r": [],
}
for image_name in tqdm(image_names):
    img = cv.imread(f"{image_folder_path}/{image_name}")
    average = img.mean(axis=0).mean(axis=0)
    mean_arrs["b"].append(average[0])
    mean_arrs["g"].append(average[1])
    mean_arrs["r"].append(average[2])

means = {k: np.mean(v) for k, v in mean_arrs.items()}
std_devs = {k: np.std(v) for k, v in mean_arrs.items()}

print(means)
print(std_devs)
