import pandas as pd
from tqdm.contrib.itertools import product

from paths import DATA_DIR
from src.features.global_features.hog_feature import HOGFeature

image_folder_path = f"{DATA_DIR}/extracted_images/face_images"

resize_size_space = [(64, 64)]
orientations_space = [6, 9]
pixels_per_cell_space = [(8, 8), (16, 16)]
cells_per_block_space = [(2, 2), (3, 3)]
block_norm_space = ["L1", "L1-sqrt", "L2", "L2-Hys"]

configs_df = pd.DataFrame(
    columns=[
        "resize_size",
        "orientations",
        "pixels_per_cell",
        "cells_per_block",
        "block_norm",
    ]
)

for i, (
    resize_size,
    orientations,
    pixels_per_cell,
    cells_per_block,
    block_norm,
) in enumerate(
    product(
        resize_size_space,
        orientations_space,
        pixels_per_cell_space,
        cells_per_block_space,
        block_norm_space,
    )
):
    hog = HOGFeature(
        resize_size=resize_size,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
    )
    hog.extract_features(image_folder_path=image_folder_path)
    hog.save_features(f"{DATA_DIR}/hog/run_{i}")

    configs_df.loc[i] = {
        "resize_size": resize_size,
        "orientations": orientations,
        "pixels_per_cell": pixels_per_cell,
        "cells_per_block": cells_per_block,
        "block_norm": block_norm,
    }

configs_df.to_csv(f"{DATA_DIR}/hog/configs.csv", index=False)
