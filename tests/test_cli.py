from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from steer.cli import _comparison_view
from steer.steering import Comparison


def comparison() -> Comparison:
    return SimpleNamespace(
        baseline="A plain answer with [brackets].",
        steered="A noticeably different answer.",
        artifact=SimpleNamespace(
            metadata=SimpleNamespace(
                concept="gym bro",
                model="example/model",
                layers=(2, 4),
            )
        ),
        strength=0.2,
        settings=SimpleNamespace(seed=42, temperature=0.7),
        device="cpu",
    )


def test_comparison_view_renders_metadata_and_generations_side_by_side():
    output = StringIO()
    console = Console(file=output, width=100, color_system=None)

    console.print(_comparison_view(comparison(), console.size.width))

    rendered = output.getvalue()
    assert "Comparison" in rendered
    assert "gym bro" in rendered
    assert "example/model" in rendered
    assert "Baseline" in rendered and "Steered" in rendered
    assert "A plain answer with [brackets]." in rendered
    assert "A noticeably different answer." in rendered
    assert rendered.index("Baseline") < rendered.index("Steered")


def test_comparison_view_stacks_generations_on_narrow_terminals():
    output = StringIO()
    console = Console(file=output, width=60, color_system=None)

    console.print(_comparison_view(comparison(), console.size.width))

    lines = output.getvalue().splitlines()
    baseline_line = next(index for index, line in enumerate(lines) if "Baseline" in line)
    steered_line = next(index for index, line in enumerate(lines) if "Steered" in line)
    assert steered_line > baseline_line + 1
