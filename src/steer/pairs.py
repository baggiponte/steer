from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

NEUTRAL_REFERENCE = "a neutral, clear, matter-of-fact assistant"


@dataclass(frozen=True)
class ContrastPair:
    concept: str
    neutral: str

    def __post_init__(self) -> None:
        if not self.concept.strip() or not self.neutral.strip():
            raise ValueError("Both sides of a contrast pair must contain text.")


def load_pairs(path: Path) -> list[ContrastPair]:
    pairs: list[ContrastPair] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read pair file {path}: {error}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            pairs.append(ContrastPair(value["concept"], value["neutral"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid contrast pair on line {line_number} of {path}: {error}"
            ) from error
    if not pairs:
        raise ValueError(f"Pair file {path} contains no contrast pairs.")
    return pairs


def save_pairs(pairs: Iterable[ContrastPair], path: Path) -> None:
    rows = [
        json.dumps(
            {"concept": pair.concept, "neutral": pair.neutral}, ensure_ascii=False
        )
        for pair in pairs
    ]
    if not rows:
        raise ValueError("Cannot save an empty contrast-pair dataset.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def pair_generation_prompt(concept: str, number_of_pairs: int) -> str:
    return f"""Create {number_of_pairs} matched contrast pairs for activation steering.

The target concept is: {concept}
The fixed reference is: {NEUTRAL_REFERENCE}

For each pair, express exactly the same underlying message twice:
- `concept`: strongly embodies the target concept
- `neutral`: sounds like the fixed reference and does not mention the concept

Vary the underlying subjects and do not discuss this task. Return only a JSON array
of objects with exactly two string fields named `concept` and `neutral`.
"""


def parse_generated_pairs(text: str, expected_count: int) -> list[ContrastPair]:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("The model did not return a JSON array of contrast pairs.")
    try:
        values = json.loads(text[start : end + 1])
        pairs = [ContrastPair(value["concept"], value["neutral"]) for value in values]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"The model returned invalid contrast-pair JSON: {error}"
        ) from error
    if len(pairs) != expected_count:
        raise ValueError(
            f"The model returned {len(pairs)} pairs; expected {expected_count}. Try again "
            "or edit the generated JSONL file manually."
        )
    return pairs
