import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

from paths import DATA_DIR

faces_path = f"{DATA_DIR}/labelled_faces/to_be_labelled"


def on_press(event):
    global img_path
    global ids
    global labels

    # Get key
    key = event.key

    # Get image name
    path = img_path[0]
    id = path.split("/")[-1]

    if key not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "x"]:
        print("Invalid Key")
    else:
        ids.append(id)
        labels.append(key)

        # Clear plot
        plt.clf()

        if len(img_path) > 0:
            img_path = img_path[1:]

            # Display next image with count
            path = img_path[0]
            img = mpimg.imread(path)
            plt.text(0, 15, len(img_path), color="r", size=20)

            plt.imshow(img)
            plt.show()


ids = []
labels = []

# Get all image paths
img_path = sorted(glob.glob(faces_path + "/*.png"))

fig = plt.figure(figsize=(5, 5))

# Load first image and add count
path = img_path[0]
img = mpimg.imread(path)
plt.text(0, 15, len(img_path), color="r", size=20)

# Add an interactive widget to figure
cid = fig.canvas.mpl_connect("key_press_event", on_press)

plt.imshow(img)
plt.show()

# Save labels
labels_df = pd.DataFrame(list(zip(ids, labels)), columns=["Name", "Label"])
labels_df.to_csv(f"{faces_path}/labels.csv", index=False)
