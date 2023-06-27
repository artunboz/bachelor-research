import matplotlib.pyplot as plt
import seaborn as sns

from paths import ROOT_DIR

names = ['Random', 'ORB', 'SimCLR', 'LBP', 'HOG', 'Color Hist.']
values = [0.114, 0.214, 0.399, 0.413, 0.426, 0.600]

palette = sns.color_palette('crest')

plt.figure(figsize=(6.4, 4))
plt.bar(names, values, color=palette)

plt.xlabel('Feature Extraction Method')
plt.ylabel('F1 Score')
plt.title('A Comparison of Feature Extraction Methods Based on K-Means++')

# plt.show()
plt.savefig(f"{ROOT_DIR}/figures/feature_f1.svg")
