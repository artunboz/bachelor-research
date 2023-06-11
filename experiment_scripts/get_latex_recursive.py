import os
import pathlib
from argparse import ArgumentParser

import pandas as pd

from paths import DATA_DIR

parser = ArgumentParser()
parser.add_argument("--root-folder", dest="root_folder")
args = parser.parse_args()

path = f"{DATA_DIR}/{args.root_folder}"


def format_run_name(run_name):
    return run_name.replace("run_", "conf. ")


def format_col_name(col_name):
    return col_name.replace("_", " ")


def save_latex(file_path):
    df = pd.read_csv(file_path)

    file = pathlib.Path(file_path)
    file_name = file.name.split(".")[0]
    folder_name = str(file.parent)
    latex_file_path = f"{folder_name}/{file_name}_latex.txt"

    with open(latex_file_path, mode="w") as f:
        df.to_latex(
            f,
            float_format="%.2f",
            index=False,
            header=[
                "configuration",
                *[
                    format_col_name(col_name)
                    for col_name in df.columns
                    if col_name != "name"
                ],
            ],
            formatters={"name": format_run_name},
        )


for root, _, files in os.walk(path):
    if len(files) == 0:
        continue

    for file_name in files:
        if file_name.endswith(".csv"):
            save_latex(f"{root}/{file_name}")
