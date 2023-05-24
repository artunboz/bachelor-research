from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras import layers


class DeepAutoencoder(Model):
    def __init__(self, layer_dims: list[int], output_dim: int) -> None:
        """Inits a DeepAutoencoder instance.

        :param layer_dims: A list of integers containing the number of neurons per
            neuron from the first layer to the bottleneck layer.
        :param output_dim: An integer indicating the output dimension of the decoder.
        """
        super().__init__(name="autoencoder")
        self.encoder: Sequential = _get_encoder_layer(layer_dims)
        self.decoder: Sequential = _get_decoder_layer(layer_dims, output_dim)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


def _get_encoder_layer(layer_dims: list[int]) -> Sequential:
    encoder_layers: list[layers.Dense] = [
        layers.Dense(d, activation="relu") for d in layer_dims
    ]
    return Sequential(encoder_layers, name="encoder")


def _get_decoder_layer(layer_dims: list[int], output_dim) -> Sequential:
    decoder_layers: list[layers.Dense] = [
        layers.Dense(d, activation="relu") for d in layer_dims[-2::-1]
    ]
    decoder_layers.append(layers.Dense(output_dim, activation="sigmoid"))
    return Sequential(decoder_layers, name="decoder")
