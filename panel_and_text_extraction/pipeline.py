"""This file contains a pipeline to extract panels from Dilbert comics, crop the texts
from the panels, and save them.
"""

import os

from paths import DATA_DIR
from panel_and_text_extraction.panel_extractor import PanelExtractor
from panel_and_text_extraction.text_cropper import TextCropper

input_folder_path = f"{DATA_DIR}/dilbert_comics_original"
output_folder_path = f"{DATA_DIR}/cleaned_panels"

if not os.path.exists(input_folder_path):
    raise ValueError(f"Could not find {input_folder_path}.")

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Construct the image paths list
input_image_paths = []
for subdir, _, files in os.walk(input_folder_path):
    if len(files) == 0:
        continue
    input_image_paths.extend([f"{subdir}/{f}" for f in files])

# Extract the panels
panel_extractor = PanelExtractor()
panel_extractor.extract_and_save_panels(input_image_paths, output_folder_path)

# Crop the text blocks from the panels
text_cropper = TextCropper()
text_cropper.crop_text_and_save_panels(output_folder_path)
