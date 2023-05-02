"""Converts the images found in the input folder to png format.
"""

import os
import re

from PIL import Image
from tqdm import tqdm

from paths import DATA_DIR

input_folder_path = f"{DATA_DIR}/dilbert_comics_gif"
output_folder_path = f"{DATA_DIR}/dilbert_comics_png"

if not os.path.exists(input_folder_path):
    raise ValueError(f"Could not find {input_folder_path}.")

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for year in tqdm(sorted(os.listdir(input_folder_path)), desc="Years"):
    input_year_dir = f"{input_folder_path}/{year}"
    output_year_dir = f"{output_folder_path}/{year}"
    os.makedirs(output_year_dir)
    for image_file in sorted(os.listdir(f"{input_year_dir}")):
        comic_date = re.search(r"(\d{4}-\d{2}-\d{2})", image_file).group()
        im = Image.open(f"{input_year_dir}/{image_file}")
        im.save(f"{output_year_dir}/{comic_date}.png", "PNG")

# Sanity check
_, _, old_files = next(os.walk(input_folder_path))
_, _, new_files = next(os.walk(output_folder_path))
assert len(old_files) == len(new_files)
