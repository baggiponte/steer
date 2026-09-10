# Activation Steering

This project demonstrates how a language model's generated behavior can be
changed by adding an extracted concept direction to its internal activations.

## Language

**Concept**:
A user-named behavior, style, or topic toward which generation should be
steered. It is contrasted against the project's fixed neutral reference.
_Avoid_: Persona, behavior vector

**Contrast pair**:
Two matched texts expressing the same underlying content, one invoking the
concept and one using the neutral reference.
_Avoid_: Training pair, positive/negative sample

**Concept artifact**:
A model-specific result containing the steering vectors extracted for a
concept at selected layers, together with the information required to use them
correctly.
_Avoid_: Model, adapter, checkpoint

**Intervention**:
The inference-time addition of a concept artifact's vectors to their selected
model layers with a signed strength.
_Avoid_: Training, fine-tuning

**Comparison**:
A baseline generation and an intervened generation produced from the same
prompt and generation settings.
_Avoid_: Benchmark, evaluation
