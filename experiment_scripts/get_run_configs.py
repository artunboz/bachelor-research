import json
import os
from argparse import ArgumentParser

import pandas as pd

from paths import DATA_DIR

parser = ArgumentParser()
parser.add_argument("feature")
args = parser.parse_args()

root_folder = f"{DATA_DIR}/{args.feature}"
files = []
for run in os.listdir(root_folder):
    if run.startswith("run_"):
        files.append(f"{root_folder}/{run}/feature_config.json")

combined_results = []
for file in files:
    with open(file, mode="r") as f:
        combined_results.append(json.load(f)[0])

configs_df = pd.DataFrame(combined_results)
configs_df.to_csv(f"{root_folder}/results/configs.csv")
