from paths import DATA_DIR
from src.features.global_features.random_feature import RandomFeature

image_folder_path = f"{DATA_DIR}/extracted_images/face_images"
n_dims = 10

rand_feat = RandomFeature(n_dims=n_dims)
rand_feat.extract_features(image_folder_path=image_folder_path)
rand_feat.save_features(f"{DATA_DIR}/random/run_0")
