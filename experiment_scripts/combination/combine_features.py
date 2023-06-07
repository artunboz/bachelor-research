from argparse import ArgumentParser

from paths import DATA_DIR
from src.features.combine_features import combine_and_save

parser = ArgumentParser()
parser.add_argument("--feature-names", nargs="+", dest="feature_names")
parser.add_argument("--feature-paths", nargs="+", dest="feature_paths")
parser.add_argument("--save-folder", dest="save_folder")
args = parser.parse_args()

feature_folder_paths = dict(zip(args.feature_names, args.feature_paths))
feature_folder_paths = {k: f"{DATA_DIR}/{v}" for k, v in feature_folder_paths.items()}
combine_and_save(feature_folder_paths, f"{DATA_DIR}/combinations/{args.save_folder}")
