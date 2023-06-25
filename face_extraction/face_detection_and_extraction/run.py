from face_extraction.face_detection_and_extraction.detector import Detector
from face_extraction.face_detection_and_extraction.extractor import extract_images

image_folder_path = "/home/artun/Desktop/research_project/research/data/cleaned_panels/panels"

detector = Detector(
    "/home/artun/Desktop/research_project/DASS_Det_Inference/data/weights/xl/xl_mixdata_finetuned_stage3.pth",
    "xl",
)

boxes = detector.predict_images(image_folder_path, nms_thold=0.45, conf_thold=0.7)
boxes.to_csv(f"/home/artun/Desktop/research_project/research/data/cleaned_panels/mk_2/boxes.csv")

extract_images("/home/artun/Desktop/research_project/research/data/cleaned_panels/panels")
