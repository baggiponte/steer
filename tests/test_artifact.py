from pathlib import Path

import torch

from steer.artifact import ArtifactMetadata, ConceptArtifact


def metadata() -> ArtifactMetadata:
    return ArtifactMetadata(
        concept="gym bro",
        model="example/model",
        model_revision="abc123",
        layers=(2, 4),
        pair_file_sha256="deadbeef",
        number_of_pairs=2,
        neutral_reference="neutral",
        extraction_token="last",
        aggregation="mean",
        torch_version="test",
        transformers_version="test",
    )


def test_artifact_round_trip(tmp_path: Path):
    artifact = ConceptArtifact(
        metadata(), {2: torch.tensor([1.0, 2.0]), 4: torch.tensor([3.0, 4.0])},
    )
    path = tmp_path / "concept.safetensors"

    artifact.save(path)

    loaded = ConceptArtifact.load(path)
    assert loaded.metadata == artifact.metadata
    assert torch.equal(loaded.vectors[2], artifact.vectors[2])
    assert torch.equal(loaded.vectors[4], artifact.vectors[4])


def test_artifact_rejects_missing_layer_vector():
    try:
        ConceptArtifact(metadata(), {2: torch.tensor([1.0])})
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("Expected invalid artifact to be rejected")
