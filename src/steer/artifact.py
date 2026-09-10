from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT_VERSION = 1
METADATA_KEY = "steer.metadata"


@dataclass(frozen=True)
class ArtifactMetadata:
    concept: str
    model: str
    model_revision: str
    layers: tuple[int, ...]
    pair_file_sha256: str
    number_of_pairs: int
    neutral_reference: str
    extraction_token: str
    aggregation: str
    torch_version: str
    transformers_version: str

    @classmethod
    def from_json(cls, value: str) -> ArtifactMetadata:
        data = json.loads(value)
        data["layers"] = tuple(data["layers"])
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True)
class ConceptArtifact:
    metadata: ArtifactMetadata
    vectors: dict[int, torch.Tensor]

    def __post_init__(self) -> None:
        expected = set(self.metadata.layers)
        if set(self.vectors) != expected:
            raise ValueError("Artifact layers and stored vectors do not match.")
        for layer, vector in self.vectors.items():
            if vector.ndim != 1:
                raise ValueError(f"Vector for layer {layer} must be one-dimensional.")
            if not torch.isfinite(vector).all():
                raise ValueError(
                    f"Vector for layer {layer} contains non-finite values."
                )

    @property
    def norms(self) -> dict[int, float]:
        return {
            layer: vector.float().norm().item()
            for layer, vector in self.vectors.items()
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            f"layer.{layer}": vector.detach().cpu().contiguous()
            for layer, vector in self.vectors.items()
        }
        metadata = {
            "format_version": str(FORMAT_VERSION),
            METADATA_KEY: self.metadata.to_json(),
        }
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(file_descriptor)
        try:
            save_file(tensors, temporary_name, metadata=metadata)
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @classmethod
    def load(cls, path: Path) -> ConceptArtifact:
        try:
            with safe_open(path, framework="pt", device="cpu") as file:
                file_metadata = file.metadata()
                if file_metadata.get("format_version") != str(FORMAT_VERSION):
                    raise ValueError("Unsupported concept-artifact format version.")
                encoded_metadata = file_metadata.get(METADATA_KEY)
                if encoded_metadata is None:
                    raise ValueError("Concept artifact is missing its metadata.")
                metadata = ArtifactMetadata.from_json(encoded_metadata)
                vectors = {
                    int(key.removeprefix("layer.")): file.get_tensor(key)
                    for key in file.keys()
                    if key.startswith("layer.")
                }
        except OSError as error:
            raise ValueError(
                f"Could not read concept artifact {path}: {error}"
            ) from error
        return cls(metadata, vectors)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
