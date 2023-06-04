import pandas as pd
from tqdm.contrib.itertools import product

from paths import DATA_DIR
from src.features.local_features.orb_feature import ORBFeature

image_folder_path = f"{DATA_DIR}/extracted_images/face_images"

resize_size_space = [(128, 128), (256, 256)]
quantization_method_space = ["fisher"]
n_components_space_space = [[10], [20]]
n_features_space = [10, 20]
scale_factor_space = [1.2]  # default
n_levels_space = [8]  # default
first_level_space = [0]  # default
wta_k_space = [2]  # default
patch_size_space = [31]  # default
fast_threshold_space = [20]  # default

configs_df = pd.DataFrame(
    columns=[
        "name",
        "resize_size",
        "quantization_method",
        "n_components_space",
        "n_features",
        # "scale_factor",
        # "n_levels",
        # "first_level",
        # "wta_k",
        # "patch_size",
        # "fast_threshold",
    ]
)

for i, (
    resize_size,
    quantization_method,
    n_components_space,
    n_features,
    scale_factor,
    n_levels,
    first_level,
    wta_k,
    patch_size,
    fast_threshold,
) in enumerate(
    product(
        resize_size_space,
        quantization_method_space,
        n_components_space_space,
        n_features_space,
        scale_factor_space,
        n_levels_space,
        first_level_space,
        wta_k_space,
        patch_size_space,
        fast_threshold_space,
    )
):
    orb = ORBFeature(
        resize_size=resize_size,
        quantization_method=quantization_method,
        n_components_space=n_components_space,
        n_features=n_features,
        scale_factor=scale_factor,
        n_levels=n_levels,
        first_level=first_level,
        wta_k=wta_k,
        patch_size=patch_size,
        fast_threshold=fast_threshold,
    )
    orb.extract_features(image_folder_path=image_folder_path)
    orb.save_features(f"{DATA_DIR}/orb_fisher/run_{i}")

    configs_df.loc[i] = {
        "name": f"run_{i}",
        "resize_size": resize_size,
        "quantization_method": quantization_method,
        "n_components_space": n_components_space,
        "n_features": n_features,
        # "scale_factor": scale_factor,
        # "n_levels": n_levels,
        # "first_level": first_level,
        # "wta_k": wta_k,
        # "patch_size": patch_size,
        # "fast_threshold": fast_threshold,
    }

configs_df.to_csv(f"{DATA_DIR}/orb_fisher/configs.csv", index=False)
