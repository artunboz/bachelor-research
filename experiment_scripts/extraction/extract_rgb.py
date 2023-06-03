import pandas as pd
from tqdm.contrib.itertools import product

from paths import DATA_DIR
from src.features.global_features.rgb_histogram_feature import RGBHistogramFeature

image_folder_path = f"{DATA_DIR}/extracted_images/face_images"

resize_size_space = [(48, 48)]
hist_size_space = [32, 64, 128, 256]

configs_df = pd.DataFrame(columns=["name", "resize_size", "hist_size"])

for i, (resize_size, hist_size) in enumerate(
    product(resize_size_space, hist_size_space)
):
    rgb_hist = RGBHistogramFeature(resize_size, hist_size)
    rgb_hist.extract_features(image_folder_path=image_folder_path)
    rgb_hist.save_features(f"{DATA_DIR}/rgb/run_{i}")

    configs_df.loc[i] = {
        "name": f"run_{i}",
        "resize_size": resize_size,
        "hist_size": hist_size,
    }

configs_df.to_csv(f"{DATA_DIR}/rgb/configs.csv", index=False)
