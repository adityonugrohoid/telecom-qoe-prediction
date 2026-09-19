"""
Export chosen image outputs from an executed notebook as PNG files.

Reads the notebook JSON directly (no re-execution) and writes the
image/png output of specific code cells to disk. Fails with a clear
error if a requested cell index is out of range, is not a code cell,
or has no image/png output, so a missing figure is never skipped
silently.

Usage:
    uv run python scripts/export_notebook_figures.py \\
        notebooks/04_qoe_prediction.ipynb docs/figures \\
        --cell 33:predicted-vs-actual-mos --cell 38:shap-summary
"""

import argparse
import base64
import json
from pathlib import Path


def load_notebook(notebook_path: Path) -> dict:
    """Load a Jupyter notebook as a dictionary.

    Args:
        notebook_path: Path to the .ipynb file.

    Returns:
        The parsed notebook JSON.
    """
    with open(notebook_path) as f:
        return json.load(f)


def extract_cell_png(notebook: dict, cell_index: int) -> bytes:
    """Extract the image/png output of one code cell.

    Args:
        notebook: The parsed notebook JSON.
        cell_index: Index of the cell in the notebook's cell list.

    Returns:
        Decoded PNG bytes.

    Raises:
        ValueError: If the cell index is out of range, the cell is not
            a code cell, or the cell has no image/png output.
    """
    cells = notebook["cells"]
    if cell_index < 0 or cell_index >= len(cells):
        raise ValueError(f"Cell index {cell_index} is out of range (0-{len(cells) - 1}).")

    cell = cells[cell_index]
    if cell["cell_type"] != "code":
        raise ValueError(f"Cell {cell_index} is a {cell['cell_type']} cell, not a code cell.")

    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if "image/png" in data:
            png_b64 = data["image/png"]
            if isinstance(png_b64, list):
                png_b64 = "".join(png_b64)
            return base64.b64decode(png_b64)

    raise ValueError(f"Cell {cell_index} has no image/png output.")


def parse_cell_spec(spec: str) -> tuple[int, str]:
    """Parse a "cell_index:output_name" spec.

    Args:
        spec: A string of the form "33:predicted-vs-actual-mos".

    Returns:
        Tuple of (cell_index, output_name).

    Raises:
        ValueError: If the spec is not in "index:name" form.
    """
    if ":" not in spec:
        raise ValueError(f"Cell spec '{spec}' must be in 'index:name' form.")
    index_str, name = spec.split(":", 1)
    if not name:
        raise ValueError(f"Cell spec '{spec}' is missing an output name.")
    return int(index_str), name


def export_figures(notebook_path: Path, output_dir: Path, cell_specs: list[str]) -> list[Path]:
    """Export the requested cell image outputs as PNG files.

    Args:
        notebook_path: Path to the executed .ipynb file.
        output_dir: Directory to write the PNG files into.
        cell_specs: List of "cell_index:output_name" strings.

    Returns:
        List of paths written.

    Raises:
        ValueError: If any requested cell or image is missing.
    """
    notebook = load_notebook(notebook_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for spec in cell_specs:
        cell_index, name = parse_cell_spec(spec)
        png_bytes = extract_cell_png(notebook, cell_index)
        output_path = output_dir / f"{name}.png"
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        written.append(output_path)
        print(f"Wrote cell {cell_index} -> {output_path} ({len(png_bytes):,} bytes)")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook_path", type=Path, help="Path to the executed .ipynb file")
    parser.add_argument("output_dir", type=Path, help="Directory to write PNG files into")
    parser.add_argument(
        "--cell",
        dest="cell_specs",
        action="append",
        required=True,
        help="Cell to export, as 'cell_index:output_name'. Repeatable.",
    )
    args = parser.parse_args()

    export_figures(args.notebook_path, args.output_dir, args.cell_specs)


if __name__ == "__main__":
    main()
