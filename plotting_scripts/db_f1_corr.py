import matplotlib.pyplot as plt
import numpy as np

db = [2.435, 6.532, 1.188, 1.243, 2.600]
f1 = [0.426, 0.413, 0.214, 0.600, 0.399]
labels = ['HOG', 'LBP', 'ORB', 'Color Hist.', 'SimCLR']
colors = ['red', 'green', 'blue', 'orange', 'purple']

plt.scatter(db, f1, c=colors)
plt.xlabel("Davies-Bouldin Score")
plt.ylabel("F1 Score")
plt.title("Correlation of Davies-Bouldin and F1")

handles = [plt.Line2D([], [], marker="o", color=c, linestyle='None') for c in colors]
plt.legend(handles, labels)

correlation = np.corrcoef(db, f1)[0, 1]
print("Correlation:", correlation)

plt.show()
