import torch

from steer.model import LoadedModel


class FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBase()


def loaded_model() -> LoadedModel:
    return LoadedModel("fake", FakeModel(), object(), torch.device("cpu"), "test")


def test_intervention_changes_decode_tokens_and_restores_hook():
    loaded = loaded_model()
    vector = torch.tensor([1.0, 2.0])
    decode = torch.zeros((1, 1, 2))
    prefill = torch.zeros((1, 3, 2))

    with loaded.intervene({0: vector}, strength=0.5):
        assert torch.equal(loaded.layers[0](decode), torch.tensor([[[0.5, 1.0]]]))
        assert torch.equal(loaded.layers[0](prefill), prefill)

    assert torch.equal(loaded.layers[0](decode), decode)


def test_validate_layers_rejects_duplicates():
    try:
        loaded_model().validate_layers((1, 1))
    except ValueError as error:
        assert "only once" in str(error)
    else:
        raise AssertionError("Expected duplicate layers to be rejected")
