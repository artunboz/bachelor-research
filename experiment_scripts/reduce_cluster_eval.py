import os
from argparse import ArgumentParser

from paths import SCRIPTS_DIR

parser = ArgumentParser()
parser.add_argument("--reduction")
parser.add_argument("--feature-path-original", dest="feature_path_original")
parser.add_argument("--feature-path-reduced", dest="feature_path_reduced")
args = parser.parse_args()

os.system(
    f"python {SCRIPTS_DIR}/reduction/reduce_{args.reduction}.py --feature-path {args.feature_path_original}"
)
os.system(
    f"python {SCRIPTS_DIR}/clustering/cluster_kmeans.py --feature-path {args.feature_path_reduced}"
)
os.system(
    f"python {SCRIPTS_DIR}/eval_and_agg.py --feature-path {args.feature_path_reduced} --n-runs 1"
)
