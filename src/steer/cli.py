from __future__ import annotations

from pathlib import Path

import click
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from steer.steering import Comparison, compare_generations, create_pairs, extract_concept


def _available_output(_context, _parameter, value: Path | None) -> Path | None:
    if value is not None and value.exists():
        raise click.BadParameter(f"{value} already exists; choose another output path.")
    return value


def _layers(_context, _parameter, value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise click.BadParameter("layers must be comma-separated integers") from error
    if not layers:
        raise click.BadParameter("select at least one layer")
    return layers


def _comparison_view(result: Comparison, width: int) -> Group:
    """Build a readable comparison that adapts to the terminal width."""
    metadata = result.artifact.metadata

    details = Table.grid(expand=True, padding=(0, 1))
    details.add_column(style="bold cyan", no_wrap=True)
    details.add_column(ratio=1)
    if width >= 80:
        details.add_column(style="bold cyan", no_wrap=True)
        details.add_column(ratio=1)
        details.add_row("Concept", metadata.concept, "Model", metadata.model)
        details.add_row(
            "Layers",
            ", ".join(map(str, metadata.layers)),
            "Strength",
            str(result.strength),
        )
        details.add_row(
            "Seed",
            str(result.settings.seed),
            "Temperature",
            str(result.settings.temperature),
        )
        details.add_row("Device", result.device, "", "")
    else:
        details.add_row("Concept", metadata.concept)
        details.add_row("Model", metadata.model)
        details.add_row("Layers", ", ".join(map(str, metadata.layers)))
        details.add_row("Strength", str(result.strength))
        details.add_row("Seed", str(result.settings.seed))
        details.add_row("Temperature", str(result.settings.temperature))
        details.add_row("Device", result.device)

    if width >= 80:
        generations = Table(
            box=box.ROUNDED,
            expand=True,
            show_header=True,
            header_style="bold",
            padding=(0, 1),
        )
        generations.add_column("Baseline", ratio=1, style="white")
        generations.add_column("Steered", ratio=1, style="white")
        generations.add_row(Text(result.baseline), Text(result.steered))
    else:
        generations = Group(
            Panel(Text(result.baseline), title="Baseline", title_align="left"),
            Panel(Text(result.steered), title="Steered", title_align="left"),
        )

    return Group(
        Panel(details, title="Comparison", title_align="left", border_style="dim"),
        Text(),
        generations,
    )


@click.group()
def main() -> None:
    """Explore activation steering in a local language model."""


@main.command("pairs")
@click.option(
    "--concept",
    required=True,
    help="Concept the model should strongly express.",
)
@click.option(
    "--model",
    "model_identifier",
    required=True,
    help="Hugging Face model ID.",
)
@click.option("--number-of-pairs", type=click.IntRange(min=1), required=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    callback=_available_output,
)
def pairs_command(
    concept: str, model_identifier: str, number_of_pairs: int, output: Path
) -> None:
    """Generate an editable contrast-pair dataset."""
    try:
        pairs = create_pairs(concept, model_identifier, number_of_pairs, output)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Wrote {len(pairs)} contrast pairs to {output}")


@main.command("extract")
@click.option("--concept", required=True, help="Name stored in the concept artifact.")
@click.option(
    "--pairs",
    "pair_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--model",
    "model_identifier",
    required=True,
    help="Hugging Face model ID.",
)
@click.option(
    "--layers",
    required=True,
    callback=_layers,
    help="Comma-separated layer numbers.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    callback=_available_output,
)
def extract_command(
    concept: str,
    pair_path: Path,
    model_identifier: str,
    layers: tuple[int, ...],
    output: Path,
) -> None:
    """Extract raw, layer-specific vectors from contrast pairs."""
    try:
        artifact = extract_concept(concept, pair_path, model_identifier, layers, output)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Wrote concept artifact to {output}")
    for layer, norm in artifact.norms.items():
        click.echo(f"  layer {layer}: norm {norm:.4f}")


@main.command("compare")
@click.option(
    "--artifact",
    "artifact_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--strength",
    type=float,
    required=True,
    help="Signed inference-time multiplier.",
)
@click.option("--prompt", required=True, help="Prompt used for both generations.")
def compare_command(artifact_path: Path, strength: float, prompt: str) -> None:
    """Compare baseline and activation-steered generations."""
    try:
        result = compare_generations(artifact_path, prompt, strength)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    console = Console()
    console.print(_comparison_view(result, console.size.width))


if __name__ == "__main__":
    main()
