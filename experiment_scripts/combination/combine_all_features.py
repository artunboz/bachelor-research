from itertools import combinations

from paths import DATA_DIR
from src.features.combine_features import combine_and_save

feature_names = ["hog", "lbp", "orb", "rgb"]
feature_paths = ["hog/run_15", "lbp/run_7", "orb_fisher/run_5", "rgb/run_2"]
feature_info_dict = dict(zip(feature_names, feature_paths))
feature_info_dict = {k: f"{DATA_DIR}/{v}" for k, v in feature_info_dict.items()}

r = 2

for features in combinations(feature_names, r):
    features_dict = {f: feature_info_dict[f] for f in features}
    save_folder_name = f"run_{'_'.join(features)}"
    combine_and_save(features_dict, f"{DATA_DIR}/combinations/{r}/{save_folder_name}")
