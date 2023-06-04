from paths import DATA_DIR
from src.features.global_features.vgg16_feature import VGG16Feature

image_folder_path = f"{DATA_DIR}/extracted_images/face_images"

vgg16 = VGG16Feature()
vgg16.extract_features(image_folder_path=image_folder_path)
vgg16.save_features(f"{DATA_DIR}/lbp/run_0")
