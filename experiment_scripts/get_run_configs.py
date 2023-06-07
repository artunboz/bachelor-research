import json
import os
from argparse import ArgumentParser

import pandas as pd

from paths import DATA_DIR

parser = ArgumentParser()
parser.add_argument("--feature")
args = parser.parse_args()

root_folder = f"{DATA_DIR}/{args.feature}"

combined_results = []
for run in sorted(os.listdir(root_folder)):
    if not run.startswith("run_"):
        continue

    with open(f"{root_folder}/{run}/feature_config.json", mode="r") as f:
        config_dict = json.load(f)
    config_dict["name"] = run
    combined_results.append(config_dict)

configs_df = pd.DataFrame(combined_results).set_index("name")
configs_df.to_csv(f"{root_folder}/results/configs.csv")
