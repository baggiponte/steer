# Vector steering for a local Mac demo

Status: research note for design discussion; no implementation is implied.

## Executive summary

The smallest convincing demo is **contrastive activation addition** on one deliberately supported model and runtime:

1. run matched concept-positive and concept-neutral texts through the same model and chat template;
2. capture the residual-stream activation at the same token position and candidate layer;
3. compute `vector = mean(positive - neutral)`;
4. during generation, replace a chosen block's residual output with `hidden + strength * vector`; and
5. show unsteered and steered answers side by side while sweeping a few layers and strengths.

This matches the essence of Ramp's public explanation and established activation-addition methods. It does not require training or changing model weights. The hard parts are not the vector subtraction; they are choosing clean contrasts, locating a useful layer, defining the intervention's token scope, calibrating strength without destroying fluency, and keeping extraction and generation exactly model-compatible.

For a solo CLI project, support one architecture first, use a transparent mean-difference extractor, and treat automatic layer selection, learned probes, multi-vector composition, a server, and a GUI as later experiments.

## What the literature establishes

### Activation Addition (ActAdd)

Turner et al. define activation engineering as modifying a model's activations at inference time. Their Activation Addition method forms a direction by contrasting intermediate activations elicited by prompt pairs such as “Love” and “Hate,” then adds a scaled direction during a forward pass. It needs no optimization and can produce an effect from a single contrast pair. The method was demonstrated for properties such as topic and sentiment, with experiments evaluating whether unrelated capabilities were preserved. ([paper](https://arxiv.org/abs/2308.10248))

ActAdd is the best conceptual starting point for the demo because its mechanism is visible and its extraction phase is trivial. A single pair, however, can encode accidental differences between the texts rather than a reusable concept.

### Contrastive Activation Addition (CAA)

CAA makes the estimate more robust by using many matched positive/negative examples and averaging their residual-stream differences. The archival paper studies behavioral steering, applies the intervention to generated-token positions, and explicitly treats layer and coefficient as experimental choices rather than universal constants. ([ACL paper](https://aclanthology.org/2024.acl-long.828/), [authors' code](https://github.com/nrimsky/CAA))

The authors' extraction implementation takes positive and negative activations at corresponding token positions, computes pairwise differences, and averages those differences per layer. ([vector-generation source](https://github.com/nrimsky/CAA/blob/main/generate_vectors.py))

This gives the minimal progression for this project: one pair for the first end-to-end demonstration, then a small JSONL set of matched pairs for a less brittle vector.

### Representation Engineering and control vectors

Representation Engineering is the broader umbrella: it studies population-level representations rather than individual neurons and separates **reading** a direction from **controlling** model behavior with that direction. Its baselines include supervised and unsupervised linear methods such as mean difference and PCA, and its examples cover honesty, harmlessness, utility, emotion, bias, and knowledge. Control is performed by transforming intermediate representations, including scaled linear addition. ([paper](https://arxiv.org/abs/2310.01405), [authors' repository](https://github.com/andyzoujm/representation-engineering))

The relevant lesson is that “steering vector” should not imply one mandatory extraction algorithm. Mean difference, a principal direction, or a linear-probe normal can all instantiate a direction, but they impose different data and validation requirements.

### Concept Activation Vectors (CAVs)

TCAV predates LLM activation steering. It defines a human concept through positive examples and random counterexamples, trains a linear classifier on their layer activations, and uses the classifier boundary's normal as a Concept Activation Vector. TCAV then measures a prediction's sensitivity with directional derivatives and statistical tests. ([ICML paper](https://proceedings.mlr.press/v80/kim18d.html), [Google's implementation](https://github.com/tensorflow/tcav))

TCAV is therefore an important ancestor and a possible future extractor, but it should not be used as a synonym for activation addition: TCAV is primarily a concept-testing/interpretability method; ActAdd and CAA causally perturb autoregressive generation.

### Ramp's demonstration

Ramp's first-party page describes essentially the mean-difference version: collect activations for prompts that strongly invoke a concept and for a neutral set, subtract the two averages, then add a scaled version of that vector to residual streams in middle layers at inference. Its “strength” control scales that intervention, and it explicitly shows that strong steering can overwhelm ordinary behavior. ([Ramp Labs: Steer AI](https://labs.ramp.com/steer-ai), [announcement referenced in the brief](https://x.com/RampLabs/status/2039726632886235648?s=20))

The public description supports the broad interaction model, but it should not be treated as evidence for a universally correct layer or multiplier. Those are model-, concept-, prompt-, and extraction-dependent.

## Extraction choices

These choices must be explicit because each changes what the resulting vector means.

| Choice | Minimal default | Why it matters |
|---|---|---|
| Examples | Matched positive/neutral pairs | Matching reduces nuisance differences; a neutral set alone is easier to build but more confounded. |
| Aggregation | Mean of pairwise differences | Transparent, cheap, and aligned with CAA. PCA and probes add value only once there are enough examples to validate them. |
| Token position | Same semantic endpoint in both texts | Comparing unlike positions mixes concept, syntax, and positional effects. The final non-padding token is a practical first rule when pairs are templated consistently. |
| Layer | Extract one vector per candidate layer | A direction is layer-specific; hidden spaces at different depths are not interchangeable even when dimensions match. |
| Prompt formatting | Exact generation tokenizer and chat template | Template tokens and assistant/user boundaries materially affect activations. |
| Centering | Encoded by paired subtraction | If using unpaired class means, record how each class was centered and weighted. |
| Normalization | Preserve raw vector initially; optionally expose unit-normalized mode | Raw magnitude contains dataset/layer scale but makes strength hard to compare. Unit normalization makes the knob clearer but changes the intervention semantics. |

**Recommendation/inference:** begin with raw mean-difference vectors and display their norms. Add unit normalization only as an explicit option, not a silent default. A future portable strength definition could scale a unit vector relative to the current residual norm, but that is a separate policy that requires evaluation.

## Where and when to intervene

The standard simple intervention is:

`h[layer, token] <- h[layer, token] + strength * v[layer]`

Here `h` is the residual-stream value returned by a transformer block. The layer and token set are part of the experiment, not implementation trivia.

Useful token scopes are:

- **decode only:** steer each newly generated token, matching CAA's clean behavioral-control setup;
- **prefill and decode:** steer prompt positions as well as generated positions, allowing the altered prompt state to enter downstream key/value caches; and
- **selected prompt positions:** closest to some activation-addition experiments, but requires a clear alignment rule and is less natural for an arbitrary-concept CLI.

**Recommendation/inference:** make token scope a first-class setting, but ship the first demo with `decode` as the documented default and `all` as an experiment. Do not hide this distinction behind a generic “steer” call.

### Autoregressive generation and the KV cache

MLX-LM generation performs a prompt prefill followed by repeated model calls with a per-layer cache. ([generation source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py), [cache source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py))

Consequences:

- Steering generated tokens requires the intervention wrapper to remain active on every decode call, not just prompt prefill.
- If the residual is changed after block `L`, blocks after `L` see the altered value and their cached keys/values reflect it. Blocks up to `L` do not retroactively change.
- A prefill-only steer can still affect future decoding through downstream caches, but it is not equivalent to decode-time steering.
- Reusing a prompt cache created under different steering settings is semantically unsafe. Cache identity must include model, vector, layers, strength, and token scope—or the first demo should simply avoid persisted/shared prompt caches.
- Speculative decoding complicates correctness because target and draft paths would not share identical interventions. It should be excluded initially.

## Layer and strength selection

The primary sources do not provide one transferable best layer or coefficient. ActAdd, CAA, Representation Engineering, and Ramp all treat placement and strength as choices requiring empirical calibration.

**Recommended search procedure:**

1. extract a vector for every transformer block in one pass over the examples;
2. test a sparse set of middle-layer candidates first (for example, relative depths around 40%, 55%, and 70%);
3. at each layer, sweep signed strengths including zero;
4. keep sampling settings and prompts fixed;
5. score both concept expression and damage to coherence/off-target answers; and
6. only then choose a friendly preset for the demo.

The sign must remain visible: positive strength should move toward the positive examples and negative strength should be usable as a falsification check. A monotonic effect near zero is more persuasive than one cherry-picked dramatic completion.

For multi-layer steering, adding the same layer-specific vector at several blocks compounds the perturbation. It should be a later feature with lower per-layer strengths and separate calibration, not the first default.

## Evaluation and failure modes

The demo should distinguish “interesting output” from evidence that a direction reliably represents a concept.

Minimum evaluation axes:

- **target effect:** blind human comparison, keyword/topic score for concrete concepts, or an independent classifier/judge for style and sentiment;
- **coherence:** repetition, malformed text, language switching, refusal collapse, and obvious loss of instruction following;
- **off-target preservation:** a few unrelated prompts where the desired answer should remain materially unchanged;
- **dose response:** baseline plus several positive and negative strengths;
- **generalization:** prompts not used to extract the vector; and
- **repeatability:** multiple sampling seeds or deterministic decoding for the diagnostic sweep.

Important failure modes:

- **confounded contrasts:** the vector captures wording, length, names, formatting, or sentiment correlated with the intended concept;
- **token misalignment:** subtracting different semantic positions creates a position/content mixture;
- **oversteering:** repetition, incoherence, concept obsession, broken formatting, or loss of chat behavior;
- **false layer confidence:** large vector norm or easy linear separability does not by itself establish causal usefulness;
- **non-transfer:** a vector may not transfer to unseen prompts, another chat template, another model revision, quantization, or a related model size;
- **asymmetry:** adding and subtracting a vector need not produce equal and opposite behavioral changes;
- **composition interference:** summing independently useful vectors need not preserve either behavior; and
- **evidence overclaim:** causal output changes show leverage at that intervention point, not that the model stores the human concept as one pure global direction.

## Feasibility in MLX and MLX-LM

MLX is suitable for the experiment: its neural-network modules are ordinary composable Python objects containing nested modules and arrays, and its dynamic computation model permits wrapping a block's call to observe or transform its output. MLX evaluates arrays lazily, so captured arrays intended for aggregation or persistence must be explicitly materialized. ([MLX module documentation](https://ml-explore.github.io/mlx/build/html/python/nn/module.html), [MLX lazy-evaluation guide](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html))

MLX-LM supplies model loading, tokenization/chat templates, quantized Apple-silicon inference, streaming generation, and KV-cache machinery. ([official repository](https://github.com/ml-explore/mlx-lm)) Its public generation customization includes logits processors, but logits are downstream of the transformer blocks and therefore cannot implement residual-stream steering. The model implementations expose their block stacks through model-specific structures. ([model source tree](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models))

**Recommendation/inference:** use a small model-family adapter that can:

- resolve the transformer block list;
- wrap one block while preserving its exact call/return and cache contract;
- capture a residual at extraction time; and
- add a vector at generation time.

Do not advertise generic MLX-LM model support in version one. Pick one dense decoder-only family, pin a known model/revision and MLX-LM version, and fail clearly for unsupported structures. Quantized weights are acceptable in principle because the intervention operates on runtime activations, but vectors must be extracted with the exact same model artifact used for generation.

## MLX versus PyTorch/Transformers for this demo

Both can run on an Apple-silicon GPU. MLX is designed around Apple's platform and MLX-LM provides Mac-oriented quantized inference. PyTorch's MPS backend maps tensors and modules to Metal/MPS Graph and provides GPU acceleration on Apple silicon, although Apple currently labels that backend beta. ([Apple's PyTorch-on-Mac page](https://developer.apple.com/metal/pytorch/), [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html))

| Concern | MLX + MLX-LM | PyTorch + Transformers |
|---|---|---|
| Apple-silicon practicality | Excellent fit, compact inference stack, strong quantized-model workflow. | GPU acceleration is available through MPS, but model/operator behavior on MPS needs verification. |
| Capturing activations | Requires model-family-aware wrapping or a modified forward path; MLX has no equivalent documented generic forward-hook surface. | Transformers can return per-layer hidden states, and PyTorch modules have removable forward hooks whose callbacks can replace module outputs. ([Transformers outputs](https://huggingface.co/docs/transformers/main_classes/output), [PyTorch hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)) |
| Steering during generation | Feasible, but preserving each MLX-LM block's return/cache contract is architecture-specific. | Hooking a selected decoder block is comparatively direct; the hook stays active during cached decode calls. |
| Portability | Mac-first and tied more closely to MLX-LM's model implementations. | Much broader ecosystem and easier reuse on CUDA systems; most cited steering repositories use this stack. |
| Simplicity for this project | Simple runtime, less simple intervention seam. | Heavier dependency stack, substantially simpler experimental instrumentation. |

**Recommendation/inference:** start with **PyTorch + Transformers on MPS** if the goal is to validate the abstraction and steering behavior with the least custom model plumbing. Its hooks and standard hidden-state outputs make the research mechanism easier to inspect and less coupled to one implementation. Start with **MLX + MLX-LM** if native Mac inference and an MLX-specific demonstration are themselves part of the product idea, accepting one explicit model adapter.

Given that MLX is a preference rather than a requirement, the pragmatic first design is backend-neutral at the domain level (`ConceptArtifact`, extraction request, intervention policy, experiment result), with a single PyTorch/Transformers backend first. A later MLX backend can implement the same small contract without forcing the CLI or stored experiment concepts to mirror either framework. This recommendation should be confirmed with a tiny performance/compatibility spike on the intended checkpoint before committing; primary documentation establishes availability and ergonomics, not a universal speed winner for this workload.

## Minimal staged implementation outline

No code should be written until the abstractions and CLI below are agreed.

### Stage 0 — lock one reproducible experiment

- Choose one Apple-silicon-friendly instruct model and one runtime: PyTorch/Transformers on MPS or MLX/MLX-LM.
- Define one concrete concept with 5–20 matched positive/neutral pairs; retain a separate prompt set for demonstration.
- Fix the model revision, chat template, sampling configuration, extraction token rule, and initial candidate layers.
- Write success criteria: visible target shift at a usable strength, coherent output, and no obvious collapse on two or three unrelated prompts.

### Stage 1 — prove observability without steering

- Load and tokenize through the chosen runtime.
- Resolve candidate transformer blocks through the runtime's supported mechanism.
- Capture same-position residual activations from positive/neutral examples.
- Verify shapes, materialization, determinism, and layer-specific norms.

### Stage 2 — create a reproducible vector artifact

- Compute mean pairwise differences per layer.
- Persist the arrays plus provenance: schema version, model identifier and revision, MLX-LM version, layer indices, hidden dimension, dtype, chat template identity, examples or their hashes, extraction token rule, aggregation method, and normalization.
- Reject artifacts that do not match the loaded model.

### Stage 3 — steer generation

- Wrap one target block and add its vector under an explicit token scope.
- Keep the runtime's ordinary sampling and streaming behavior.
- Disable prompt-cache reuse and speculative decoding initially.
- Guarantee restoration of the original model block after generation, including on errors.

### Stage 4 — make the CLI demonstrate causality

- One command creates a named vector artifact from a small data file.
- One command compares the same prompt at strength `0` and one or more signed strengths.
- Output reports model, vector, layer, scope, strength, and seed alongside the text.
- A sweep mode tries a small layer/strength grid and saves structured results; it need not be interactive or distributed.

### Stage 5 — validate before expanding

- Run held-out target, coherence, off-target, and dose-response checks.
- Choose one conservative default layer/strength only for the pinned demo model and concept.
- Add further extractors (PCA or linear probe), multiple simultaneous vectors, automatic selection, or a UI only after the basic experiment is stable.

## Design questions to settle next

1. Is the core object a **layer-specific vector artifact**, or a **concept artifact** containing one vector per layer? The latter fits extraction and sweeping better.
2. Should the first user workflow be two explicit phases (`extract`, then `compare`) or a single convenience `demo` command that composes them while keeping artifacts inspectable?
3. Should the default token scope follow CAA (`decode`) or Ramp's broader public description (`all`)? Both should remain expressible.
4. Is raw strength sufficient for the first model, or should the interface define a normalized strength immediately?
5. Which one model family and checkpoint define the compatibility boundary?
6. How much provenance belongs inside the artifact versus a separate experiment result file?

My recommendation is: a concept artifact with per-layer raw mean-difference vectors; explicit `extract`, `compare`, and small `sweep` commands; decode-only default; raw signed strength with vector norms shown; one pinned model family; and complete extraction provenance stored alongside the vectors.

## Primary sources

- Turner et al., [*Steering Language Models With Activation Engineering*](https://arxiv.org/abs/2308.10248).
- Rimsky et al., [*Steering Llama 2 via Contrastive Activation Addition*](https://aclanthology.org/2024.acl-long.828/) and [official code](https://github.com/nrimsky/CAA).
- Zou et al., [*Representation Engineering: A Top-Down Approach to AI Transparency*](https://arxiv.org/abs/2310.01405) and [official code](https://github.com/andyzoujm/representation-engineering).
- Kim et al., [*Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors*](https://proceedings.mlr.press/v80/kim18d.html) and [official code](https://github.com/tensorflow/tcav).
- Ramp Labs, [*Steer AI*](https://labs.ramp.com/steer-ai).
- Apple MLX team, [MLX documentation](https://ml-explore.github.io/mlx/) and [MLX-LM](https://github.com/ml-explore/mlx-lm).
