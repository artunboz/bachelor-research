import json

from tqdm.contrib.itertools import product

from paths import DATA_DIR
from src.dimensionality_reduction.autoencoder.autoencoder_reducer import (
    AutoencoderReducer,
)
from src.dimensionality_reduction.autoencoder.sparse_autoencoder import (
    SparseAutoencoder,
)

features_dir = f"{DATA_DIR}/lbp/run_46"
reductions_dir = f"{features_dir}/reductions/sparse_ae"

latent_dim_space = [128, 256, 512]
lambda_space = [0.001, 0.0001]
beta_space = [2, 3]
p_space = [0.1, 0.15]

with open(f"{features_dir}/feature_config.json", mode="r") as f:
    output_dim = json.load(f)["feature_dim"]

for i, (latent_dim, lambda_, beta, p) in enumerate(
    product(latent_dim_space, lambda_space, beta_space, p_space)
):
    ae = SparseAutoencoder(
        latent_dim=3, output_dim=output_dim, lambda_=lambda_, beta=beta, p=p
    )
    reducer = AutoencoderReducer(ae, optimizer="adam", loss="mse")
    reducer.reduce_dimensions(features_dir=features_dir, epochs=30, batch_size=256)
    reducer.save_reduced_features(f"{reductions_dir}/run_{i}")
    with open(f"{reductions_dir}/run_{i}/reducer_config.json", mode="r") as f:
        reducer_config = json.load(f)
    reducer_config["latent_dim"] = latent_dim
    reducer_config["lambda_"] = lambda_
    reducer_config["beta"] = beta
    reducer_config["p"] = p
    with open(f"{reductions_dir}/run_{i}/reducer_config.json", mode="w") as f:
        json.dump(reducer_config, f)
