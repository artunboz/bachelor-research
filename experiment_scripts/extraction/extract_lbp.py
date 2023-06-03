import pandas as pd
from tqdm.contrib.itertools import product

from paths import DATA_DIR
from src.features.global_features.lbp_feature import LBPFeature

image_folder_path = f"{DATA_DIR}/extracted_images/face_images"

resize_size_space = [(48, 48)]
r_space = [1, 2, 3]
method_space = ["default", "ror", "uniform"]

configs_df = pd.DataFrame(columns=["resize_size", "p", "r", "method"])

for i, (resize_size, r, method) in enumerate(
    product(resize_size_space, r_space, method_space)
):
    p = 8 * r
    lbp = LBPFeature(resize_size=resize_size, p=p, r=r, method=method)
    lbp.extract_features(image_folder_path=image_folder_path)
    lbp.save_features(f"{DATA_DIR}/lbp/run_{i}")

    configs_df.loc[i] = {"resize_size": resize_size, "p": p, "r": r, "method": method}

configs_df.to_csv(f"{DATA_DIR}/lbp/configs.csv", index=False)
