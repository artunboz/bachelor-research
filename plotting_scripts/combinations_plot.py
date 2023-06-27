import matplotlib.pyplot as plt
import pandas as pd

from paths import DATA_DIR, ROOT_DIR

csv_files = [
    "first_feature",
    "second_feature",
    "without_reduction",
    "pca",
    "autoencoder",
]
labels = [
    "first feature",
    "second feature",
    "combined feature and no reduction",
    "combined feature reduced with pca",
    "combined feature reduced with neural autoencoder",
]

dataframes = []
for file in csv_files:
    df = pd.read_csv(f"{DATA_DIR}/results/combinations/2/{file}.csv")
    dataframes.append(df)

row_names = df.iloc[:, 0].values
num_rows = len(row_names)

bar_width = 0.23
bar_gap = 0.1
group_gap = 2

total_width = (bar_width + bar_gap) * len(csv_files)
index = range(num_rows)

fig, ax = plt.subplots(figsize=(20, 4.5))
for i, df in enumerate(dataframes):
    values = df.iloc[:, 1].values
    x_pos = [group_gap * x + (i * (bar_width + bar_gap)) for x in index]
    ax.bar(x_pos, values, bar_width, label=labels[i])

for bars in ax.containers:
    ax.bar_label(bars)

ax.set_xticks([group_gap * x + (total_width / 2) - 0.18 for x in index])
ax.set_xticklabels(row_names)

ax.set_xlabel("Combination")
ax.set_ylabel("F1 Score")
ax.set_title("A Comparison of Feature Combinations Based on K-Means++")

ax.legend()

# plt.show()
plt.savefig(f"{ROOT_DIR}/figures/combinations_f1.svg")
