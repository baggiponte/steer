# steer

A small command-line demonstration of activation steering: extract a concept
direction from matched texts, then add it to a model's hidden state during
generation without changing the prompt or model weights.

The documented demo uses `google/gemma-3-1b-it`, PyTorch, and Hugging Face
Transformers on Apple silicon.

## Setup

```bash
uv sync --extra dev
```

Gemma requires accepting Google's usage terms on Hugging Face. Download the
model once before working offline:

```bash
hf download google/gemma-3-1b-it
```

## Try the reviewed example

Extract raw vectors at explicitly selected layers:

```bash
uv run steer extract \
  --concept "concise French" \
  --pairs examples/french.jsonl \
  --model google/gemma-3-1b-it \
  --layers 11,12 \
  --output french.safetensors
```

Compare the same prompt and sampling seed with and without the intervention:

```bash
uv run steer compare \
  --artifact french.safetensors \
  --strength 0.85 \
  --prompt "Give me three tips for keeping houseplants healthy."
```

Strength is applied only during comparison. Extraction always stores raw,
unscaled vectors, so the same artifact can be reused with different signed
strengths. At the documented strength, the baseline stays in English while the
intervened answer switches to French. A negative strength steers back toward
neutral English; larger positive values make the language shift stronger but
can reduce factual precision.

## Generate another concept dataset

```bash
uv run steer pairs \
  --concept "film noir detective" \
  --model google/gemma-3-1b-it \
  --number-of-pairs 16 \
  --output film-noir.jsonl
```

The JSONL output is intentionally editable. Review it before extraction: a
steering vector can capture every systematic difference in the pair data, not
only the concept you intended.

## Scope

This first version intentionally has no automatic layer selection, strength
calibration, normalization, sweeps, multi-concept composition, server, or GUI.
See the research note in `docs/research/vector-steering.md` for the reasoning
and limitations.
