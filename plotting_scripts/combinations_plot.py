import matplotlib.pyplot as plt
import pandas as pd

from paths import DATA_DIR

# List of CSV file names
csv_files = [
    "first_feature",
    "second_feature",
    "without_reduction",
    "pca",
    "autoencoder",
]

# Read the CSV files and store the data in a list of DataFrames
dataframes = []
for file in csv_files:
    df = pd.read_csv(f"{DATA_DIR}/results/combinations/2/{file}.csv")
    dataframes.append(df)

# Extract the row names and column values
row_names = df.iloc[:, 0].values
num_rows = len(row_names)

# Set the width of each bar and the gap between bars
bar_width = 0.23
bar_gap = 0.1
group_gap = 2

# Calculate the total width of each set of bars
total_width = (bar_width + bar_gap) * len(csv_files)
index = range(num_rows)

# Create the bar chart
fig, ax = plt.subplots(figsize=(20, 6))
for i, df in enumerate(dataframes):
    values = df.iloc[:, 1].values
    x_pos = [group_gap * x + (i * (bar_width + bar_gap)) for x in index]
    ax.bar(x_pos, values, bar_width, label=csv_files[i])

for bars in ax.containers:
    ax.bar_label(bars)

# Set the x-axis tick positions and labels
ax.set_xticks([group_gap * x + (total_width / 2) - 0.18 for x in index])
ax.set_xticklabels(row_names)

# Set the axis labels and the chart title
ax.set_xlabel("Combination")
ax.set_ylabel("F1 Score")

# Add a legend
ax.legend()

# Display the bar chart
plt.show()
