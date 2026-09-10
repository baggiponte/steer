from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers

from .artifact import ArtifactMetadata, ConceptArtifact, sha256_file
from .model import GenerationSettings, LoadedModel
from .pairs import (
    NEUTRAL_REFERENCE,
    ContrastPair,
    load_pairs,
    pair_generation_prompt,
    parse_generated_pairs,
    save_pairs,
)


@dataclass(frozen=True)
class Comparison:
    baseline: str
    steered: str
    artifact: ConceptArtifact
    strength: float
    settings: GenerationSettings
    device: str


def create_pairs(
    concept: str,
    model_identifier: str,
    number_of_pairs: int,
    output: Path,
) -> list[ContrastPair]:
    if not concept.strip():
        raise ValueError("Concept must contain text.")
    if number_of_pairs < 1:
        raise ValueError("Number of pairs must be at least one.")
    model = LoadedModel.load(model_identifier)
    settings = GenerationSettings(max_new_tokens=max(512, number_of_pairs * 96))
    response = model.generate(
        pair_generation_prompt(concept, number_of_pairs), settings
    )
    pairs = parse_generated_pairs(response, number_of_pairs)
    save_pairs(pairs, output)
    return pairs


def extract_concept(
    concept: str,
    pair_path: Path,
    model_identifier: str,
    layers: Sequence[int],
    output: Path,
) -> ConceptArtifact:
    pairs = load_pairs(pair_path)
    model = LoadedModel.load(model_identifier)
    selected = model.validate_layers(layers)
    concept_activations = model.capture([pair.concept for pair in pairs], selected)
    neutral_activations = model.capture([pair.neutral for pair in pairs], selected)
    vectors = {
        layer: (concept_activations[layer] - neutral_activations[layer]).mean(dim=0)
        for layer in selected
    }
    metadata = ArtifactMetadata(
        concept=concept,
        model=model.identifier,
        model_revision=model.revision,
        layers=selected,
        pair_file_sha256=sha256_file(pair_path),
        number_of_pairs=len(pairs),
        neutral_reference=NEUTRAL_REFERENCE,
        extraction_token="final-token-before-chat-terminator",
        aggregation="mean-pairwise-difference",
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
    )
    artifact = ConceptArtifact(metadata, vectors)
    artifact.save(output)
    return artifact


def compare_generations(
    artifact_path: Path,
    prompt: str,
    strength: float,
    settings: GenerationSettings | None = None,
) -> Comparison:
    artifact = ConceptArtifact.load(artifact_path)
    model = LoadedModel.load(artifact.metadata.model)
    model.validate_layers(artifact.metadata.layers)
    if (
        model.revision != "unknown"
        and artifact.metadata.model_revision != model.revision
    ):
        raise ValueError(
            "Concept artifact model revision does not match the loaded model revision."
        )
    settings = settings or GenerationSettings()
    baseline = model.generate(prompt, settings)
    with model.intervene(artifact.vectors, strength):
        steered = model.generate(prompt, settings)
    return Comparison(
        baseline, steered, artifact, strength, settings, str(model.device)
    )
