import copy
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from dass_det.data.data_augment import ValTransform
from dass_det.models.yolo_head import YOLOXHead
from dass_det.models.yolo_head_stem import YOLOXHeadStem
from dass_det.models.yolo_pafpn import YOLOPAFPN
from dass_det.models.yolox import YOLOX
from dass_det.utils import postprocess
from dass_det.utils.visualize import vis


class Detector:
    def __init__(
        self, model_path, model_size, transformer=None, resize_size=(1024, 1024)
    ):
        if model_size == "xs":
            depth, width = 0.33, 0.375
        elif model_size == "xl":
            depth, width = 1.33, 1.25
        else:
            raise ValueError(f"Unrecognized model_size: {model_size}")
        self.model = Detector._load_model(depth, width, model_path)

        if transformer is None:
            self.transformer = ValTransform()
        else:
            self.transformer = transformer
        self.resize_size = resize_size

    def predict_image(
        self, image_path, nms_thold=0.4, conf_thold=0.65, visualize=False
    ):
        # Prepare the image
        img = cv2.imread(image_path)
        h, w, c = img.shape
        img, _ = self.transformer(img, None, self.resize_size)
        scale = min(self.resize_size[0] / h, self.resize_size[1] / w)

        # Predict
        img_cu = torch.Tensor(copy.deepcopy(img)).unsqueeze(0).cuda()
        with torch.no_grad():
            face_preds, body_preds = self.model(img_cu, mode=0)
            face_preds = postprocess(face_preds, 1, conf_thold, nms_thold)[0]
            body_preds = postprocess(body_preds, 1, conf_thold, nms_thold)[0]

        if face_preds is not None:
            len_faces = face_preds.shape[0]
        else:
            len_faces = 0

        if body_preds is not None:
            len_bodies = body_preds.shape[0]
        else:
            len_bodies = 0

        classes = torch.cat([torch.zeros(len_faces), torch.ones(len_bodies)])

        if face_preds is not None and body_preds is not None:
            preds = torch.cat([face_preds, body_preds], dim=0)
        elif face_preds is not None:
            preds = face_preds
        elif body_preds is not None:
            preds = body_preds
        else:
            warnings.warn(f"No faces or bodies are found in {image_path}!")
            return None

        preds[:, :4] /= scale
        bboxes = preds[:, :4]
        scores = preds[:, 4]

        if visualize:
            Detector._visualize(image_path, bboxes, scores, classes)

        return bboxes, scores, classes

    def predict_images(self, image_folder_path, nms_thold=0.4, conf_thold=0.65):
        rows = []
        for image_name in tqdm(sorted(os.listdir(image_folder_path)), desc="Images"):
            image_path = f"{image_folder_path}/{image_name}"
            predictions = self.predict_image(image_path, nms_thold, conf_thold)
            if predictions is None:
                continue

            bboxes, scores, classes = predictions
            for b, s, c in zip(bboxes, scores, classes):
                rows.append(
                    [
                        image_name,
                        max(round(b[0].item()), 0),
                        max(round(b[1].item()), 0),
                        max(round(b[2].item()), 0),
                        max(round(b[3].item()), 0),
                        round(s.item(), 3),
                        int(c),
                    ]
                )
        return pd.DataFrame(
            rows, columns=["image_name", "x0", "y0", "x1", "y1", "confidence", "type"]
        )

    @staticmethod
    def _load_model(depth, width, model_path):
        model = YOLOX(
            backbone=YOLOPAFPN(depth=depth, width=width),
            head_stem=YOLOXHeadStem(width=width),
            face_head=YOLOXHead(1, width=width),
            body_head=YOLOXHead(1, width=width),
        )

        d = torch.load(model_path, map_location=torch.device("cpu"))

        if "teacher_model" in d.keys():
            model.load_state_dict(d["teacher_model"])
        else:
            model.load_state_dict(d["model"])

        model = model.eval().cuda()
        return model

    @staticmethod
    def _visualize(image_path, bboxes, scores, classes):
        p_img = cv2.imread(image_path)[:, :, ::-1]
        plt.imshow(
            vis(
                copy.deepcopy(p_img),
                bboxes,
                scores,
                classes,
                conf=0.0,
                class_names=["Face", "Body"],
            )
        )
        plt.show()

    def latent_pass(self, image_path):
        # Prepare the image
        img = cv2.imread(image_path)
        h, w, c = img.shape
        img, _ = self.transformer(img, None, self.resize_size)
        scale = min(self.resize_size[0] / h, self.resize_size[1] / w)

        # Predict
        img_cu = torch.Tensor(copy.deepcopy(img)).unsqueeze(0).cuda()
        with torch.no_grad():
            fpn_outs = self.model.backbone(img_cu)
            fpn_outs = self.model.head_stem(fpn_outs)
            foutputs = self.model.face_head(fpn_outs)
            # for i in range(3):
            #     print(foutputs[i].shape)
