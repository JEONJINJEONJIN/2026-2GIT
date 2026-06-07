from types import SimpleNamespace

from src.models.loader import get_model_layers


def test_get_model_layers_supports_base_decoder_model_path():
    layers = [object()]
    model = SimpleNamespace(layers=layers)

    assert get_model_layers(model) is layers

