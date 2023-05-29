from tqdm.contrib.itertools import product

from paths import DATA_DIR
from src.features.local_features.orb_feature import ORBFeature

image_folder_path = f"{DATA_DIR}/extracted_images/face_images"

resize_size_space = [(300, 300), (400, 400)]
quantization_method_space = ["fisher"]
n_components_space_space = [[100]]
n_features_space = [10]
scale_factor_space = [1.1, 1.2]
n_levels_space = [6, 8]
first_level_space = [0]
wta_k_space = [2]
patch_size_space = [20, 30, 40]
fast_threshold_space = [20]

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
    orb.save_features(f"{DATA_DIR}/orb_image_size_exps/run_{i}")
