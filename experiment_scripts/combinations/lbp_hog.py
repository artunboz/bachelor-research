import numpy as np
from src.dimensionality_reduction.autoencoder.deep_autoencoder import DeepAutoencoder
from src.dimensionality_reduction.autoencoder.autoencoder_reducer import (
    AutoencoderReducer,
)
from paths import DATA_DIR


lbp_folder_path = f"{DATA_DIR}/lbp/run_46"
hog_folder_path = f"{DATA_DIR}/hog/run_13"
comb_folder_path = f"{DATA_DIR}/combinations/lbp_hog"

lbp_features = np.load(f"{lbp_folder_path}/features.npy")
hog_features = np.load(f"{hog_folder_path}/features.npy")
comb_features = np.concatenate((lbp_features, hog_features), axis=1)

np.save(f"{comb_folder_path}/features.npy", comb_features)

layers_space = [[1024], [1024, 512], [1024, 512, 256]]
output_dim = comb_features.shape[1]

for i, layers in enumerate(layers_space):
    ae = DeepAutoencoder(layer_dims=layers, output_dim=output_dim)
    reducer = AutoencoderReducer(autoencoder=ae, optimizer="adam", loss="mse")
    reducer.reduce_dimensions(features_dir=comb_folder_path, epochs=30, batch_size=256)
    reducer.save_reduced_features(f"{comb_folder_path}/reductions/run_{i}")
