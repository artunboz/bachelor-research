import os
from argparse import ArgumentParser

from paths import SCRIPTS_DIR

parser = ArgumentParser()
parser.add_argument("--reduction")
parser.add_argument("--feature-path", dest="feature_path")
args = parser.parse_args()

os.system(
    f"python {SCRIPTS_DIR}/reduction/reduce_{args.reduction}.py --feature-path {args.feature_path}"
)
os.system(
    f"python {SCRIPTS_DIR}/clustering/cluster_kmeans.py --feature-path {args.feature_path}/reductions/{args.reduction}"
)
os.system(
    f"python {SCRIPTS_DIR}/eval_and_agg.py --feature-path {args.feature_path}/reductions/{args.reduction} --n-runs 1"
)
