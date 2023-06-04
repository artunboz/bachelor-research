import os
from paths import DATA_DIR
import cv2 as cv


image_folder_path = f"{DATA_DIR}/cleaned_panels/mk_2/face_images"

image_names = os.listdir(image_folder_path)
n_images = len(image_names)
height_sum = 0
width_sum = 0
for image_name in image_names:
    img = cv.imread(f"{image_folder_path}/{image_name}")
    height_sum += img.shape[0]
    width_sum += img.shape[1]

print(
    f"average height = {height_sum / n_images}, average width = {width_sum / n_images}"
)
