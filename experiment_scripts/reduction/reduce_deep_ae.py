import json

from paths import DATA_DIR
from src.dimensionality_reduction.autoencoder.autoencoder_reducer import (
    AutoencoderReducer,
)
from src.dimensionality_reduction.autoencoder.deep_autoencoder import DeepAutoencoder

features_dir = f"{DATA_DIR}/lbp/run_0"

layer_dims_space = [[128], [256, 128], [512, 256, 128], [1024, 512, 256, 128]]

with open(f"{features_dir}/feature_config.json", mode="r") as f:
    output_dim = json.load(f)["feature_dim"]

for i, layer_dims in enumerate(layer_dims_space):
    ae = DeepAutoencoder(layer_dims=layer_dims, output_dim=output_dim)
    reducer = AutoencoderReducer(ae, optimizer="adam", loss="mse")
    reducer.reduce_dimensions(features_dir=features_dir, epochs=20, batch_size=256)
    reducer.save_reduced_features(f"{features_dir}/reductions/run_{i}")
    with open(f"{features_dir}/reductions/run_{i}/reducer_config.json", mode="r") as f:
        reducer_config = json.load(f)
    reducer_config["layer_dims"] = layer_dims
    with open(f"{features_dir}/reductions/run_{i}/reducer_config.json", mode="w") as f:
        json.dump(reducer_config, f)
