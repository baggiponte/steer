from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 160
    temperature: float = 0.2
    seed: int = 42


@dataclass
class LoadedModel:
    identifier: str
    model: PreTrainedModel
    tokenizer: object
    device: torch.device
    revision: str

    @classmethod
    def load(cls, identifier: str) -> LoadedModel:
        device = _best_device()
        dtype = torch.float16 if device.type == "mps" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(identifier)
        model = AutoModelForCausalLM.from_pretrained(identifier, dtype=dtype)
        model.to(device)
        model.eval()
        revision = getattr(model.config, "_commit_hash", None) or "unknown"
        loaded = cls(identifier, model, tokenizer, device, revision)
        loaded.layers  # fail early for unsupported model structures
        return loaded

    @property
    def layers(self) -> Sequence[torch.nn.Module]:
        base_model = getattr(self.model, "model", None)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError(
                f"Model {self.identifier} does not expose decoder layers at model.layers."
            )
        return layers

    def validate_layers(self, layers: Sequence[int]) -> tuple[int, ...]:
        selected = tuple(layers)
        if not selected:
            raise ValueError("Select at least one model layer.")
        if len(set(selected)) != len(selected):
            raise ValueError("Each selected layer must appear only once.")
        invalid = [
            layer for layer in selected if layer < 0 or layer >= len(self.layers)
        ]
        if invalid:
            raise ValueError(
                f"Invalid layer(s) {invalid}; {self.identifier} has layers 0–{len(self.layers) - 1}."
            )
        return selected

    def generate(self, prompt: str, settings: GenerationSettings) -> str:
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        _seed(settings.seed)
        generation_arguments = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.temperature > 0,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if settings.temperature > 0:
            generation_arguments["temperature"] = settings.temperature
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_arguments)
        prompt_length = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(
            output[0, prompt_length:], skip_special_tokens=True
        ).strip()

    def capture(
        self, texts: Sequence[str], layers: Sequence[int]
    ) -> dict[int, torch.Tensor]:
        selected = self.validate_layers(layers)
        captured = {layer: [] for layer in selected}
        for text in texts:
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=False,
                continue_final_message=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            with torch.inference_mode():
                output = self.model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            for layer in selected:
                captured[layer].append(
                    output.hidden_states[layer + 1][0, -1].float().cpu()
                )
        return {layer: torch.stack(values) for layer, values in captured.items()}

    @contextmanager
    def intervene(
        self, vectors: Mapping[int, torch.Tensor], strength: float
    ) -> Iterator[None]:
        selected = self.validate_layers(tuple(vectors))
        handles = []

        def hook_for(vector: torch.Tensor):
            def hook(_module, _arguments, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.shape[-2] != 1:  # do not alter prompt prefill
                    return output
                steered = hidden + strength * vector.to(hidden.device, hidden.dtype)
                if isinstance(output, tuple):
                    return (steered, *output[1:])
                return steered

            return hook

        try:
            for layer in selected:
                handles.append(
                    self.layers[layer].register_forward_hook(hook_for(vectors[layer]))
                )
            yield
        finally:
            for handle in handles:
                handle.remove()


def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
