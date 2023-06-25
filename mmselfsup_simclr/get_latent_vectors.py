import os
import pickle

import cv2 as cv
import numpy as np
import torch
from mmselfsup.models.backbones.resnet import ResNet
from tqdm import tqdm

model_path = "/media/zilong/DATA1/Artun/mmselfsup/work_dirs/selfsup/simclr_resnet50_8xb32-coslr-200e_dilbert/epoch_200_backbone-weights.pth"
model = ResNet(50)
model.load_state_dict(
    torch.load(model_path, map_location=torch.device("cpu"))["state_dict"]
)
model = model.double()

images_path = "/media/zilong/DATA1/Artun/research/data/extracted_images/face_images"
mean = (198.878, 167.418, 132.772)
std = (21.34, 25.105, 26.093)


def load_preprocess_prepare(image_path):
    img = cv.imread(image_path)[:, :, ::-1].astype(float)
    for i in range(3):
        img[:, :, i] = img[:, :, i] - mean[i]
        img[:, :, i] = img[:, :, i] / std[i]

    img = cv.resize(img, (256, 256))
    center = 128
    h = 224
    y = center - h // 2
    crop_img = img[y: y + h, y: y + h, :]
    return np.transpose(crop_img, (2, 0, 1))


image_names = sorted(os.listdir(images_path))
all_features = np.empty((len(image_names), 2048))

model.eval()
with torch.no_grad():
    c = 0
    batch_size = 16
    n_runs = (len(image_names) // batch_size) + 1
    with tqdm(total=n_runs) as pbar:
        while c < len(image_names):
            batch_images = image_names[c: c + batch_size]
            cur_batch_size = len(batch_images)
            batch = np.empty((cur_batch_size, 3, 224, 224))
            for i in range(cur_batch_size):
                batch[i] = load_preprocess_prepare(f"{images_path}/{batch_images[i]}")

            batch_features = model(torch.from_numpy(batch).double())[0]
            batch_features = torch.nn.functional.avg_pool2d(batch_features, 7)
            for i, feat in enumerate(batch_features):
                feat_np = feat.ravel().numpy()
                all_features[c + i] = feat_np
            c += batch_size
            pbar.update(1)

np.save("/media/zilong/DATA1/Artun/research/data/simclr/run_1/features.npy",
        all_features)
with open(
        "/media/zilong/DATA1/Artun/research/data/simclr/image_names.pickle", mode="wb"
) as f:
    pickle.dump(image_names, f)
