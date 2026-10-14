#!/usr/bin/env python
"""
Convert Python scripts to Jupyter notebooks.

The canonical format uses VS Code / Spyder cell markers:
    # %%              → code cell boundary
    # %% [markdown]   → markdown cell boundary
                        (content follows as comment lines: # text)

Usage:
    # Batch: convert all .py files under examples_py/ → examples_ipynb/
    python convert_py_to_ipynb.py

    # Single file
    python convert_py_to_ipynb.py <input.py> <output.ipynb>

    # Check only: exit 1 if any notebook is missing or out of date
    python convert_py_to_ipynb.py --check
"""

import os
import sys
import json
from pathlib import Path


def with_newlines(lines):
    """
    Terminate every line but the last with '\\n'.

    nbformat joins a list-valued 'source' with '' (no separator), so
    lines without their terminator collapse into one line on GitHub and
    nbviewer.
    """
    return [line + "\n" for line in lines[:-1]] + lines[-1:]


def parse_py_to_cells(py_content):
    """
    Parse a Python file using '# %%' cell markers into notebook cells.

    Returns:
        list: List of dicts with 'cell_type' and 'source' keys
    """
    lines = py_content.splitlines()
    cells = []
    current_type = None
    current_source = []

    def flush():
        if current_type is None:
            return
        # Remove leading and trailing blank lines
        while current_source and current_source[0].strip() == "":
            current_source.pop(0)
        while current_source and current_source[-1].strip() == "":
            current_source.pop()
        if not current_source:
            return
        if current_type == "markdown":
            cleaned = []
            for line in current_source:
                if line.startswith("# "):
                    cleaned.append(line[2:])
                elif line.startswith("#"):
                    cleaned.append(line[1:])
                else:
                    cleaned.append(line)
            cells.append({"cell_type": "markdown", "source": with_newlines(cleaned)})
        else:
            cells.append({"cell_type": "code", "source": with_newlines(current_source)})

    for line in lines:
        stripped = line.strip()
        if (
            stripped == "# %%"
            or stripped.startswith("# %% ")
            and "[markdown]" not in stripped
        ):
            flush()
            current_type = "code"
            current_source = []
        elif stripped.startswith("# %%") and "[markdown]" in stripped:
            flush()
            current_type = "markdown"
            current_source = []
        else:
            if current_type is None:
                # Content before any marker → code cell
                current_type = "code"
            current_source.append(line)

    flush()
    return cells


def create_notebook(cells):
    """
    Create a Jupyter notebook structure from parsed cells.

    Args:
        cells: List of cell dicts with 'cell_type' and 'source' keys

    Returns:
        dict: Jupyter notebook structure
    """
    nb_cells = []

    for cell in cells:
        if cell["cell_type"] == "markdown":
            nb_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": cell["source"],
            }
        else:  # code
            nb_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": cell["source"],
                "outputs": [],
            }
        nb_cells.append(nb_cell)

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    return notebook


def render_notebook(input_path):
    """
    Convert a Python file to notebook JSON text.

    Args:
        input_path: Path to input .py file

    Returns:
        tuple: (notebook JSON text, number of cells)
    """
    # Read the Python file
    with open(input_path, "r", encoding="utf-8") as f:
        py_content = f.read()

    # Parse into cells
    cells = parse_py_to_cells(py_content)

    # Create notebook structure
    notebook = create_notebook(cells)

    return json.dumps(notebook, indent=1, ensure_ascii=False), len(cells)


def convert_file(input_path, output_path):
    """
    Convert a single Python file to a Jupyter notebook.

    Args:
        input_path: Path to input .py file
        output_path: Path to output .ipynb file
    """
    print(f"Converting {input_path.name} -> {output_path.name}")

    text, n_cells = render_notebook(input_path)

    # Write the notebook
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  ✓ Created {output_path.name} with {n_cells} cells")


def check_files(pairs):
    """
    Report notebooks that are missing or differ from their Python source.

    Args:
        pairs: List of (input .py path, output .ipynb path)

    Returns:
        int: Number of stale or missing notebooks
    """
    stale = 0
    for py_file, output_file in pairs:
        rel_py, rel_nb = os.path.relpath(py_file), os.path.relpath(output_file)
        text, _ = render_notebook(py_file)
        if not output_file.exists():
            print(f"  ✗ Missing {rel_nb} (from {rel_py})")
            stale += 1
        elif output_file.read_text(encoding="utf-8") != text:
            print(f"  ✗ Out of date {rel_nb} (from {rel_py})")
            stale += 1
    return stale


def main():
    """
    Main conversion function.
    """
    script_dir = Path(__file__).parent
    args = sys.argv[1:]
    check = "--check" in args
    args = [a for a in args if a != "--check"]

    # Single-file mode: convert_py_to_ipynb.py <input.py> <output.ipynb>
    if len(args) == 2:
        input_path = Path(args[0])
        output_path = Path(args[1])
        if check:
            sys.exit(1 if check_files([(input_path, output_path)]) else 0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        convert_file(input_path, output_path)
        return

    # Batch mode: convert all .py files under examples_py/ → examples_ipynb/,
    # mirroring subfolders (e.g. experimental/)
    input_dir = script_dir / "examples_py"
    output_dir = script_dir / "examples_ipynb"

    py_files = sorted(input_dir.rglob("*.py"))
    if not py_files:
        print("No Python files found to convert.")
        return
    pairs = [
        (p, output_dir / p.relative_to(input_dir).with_suffix(".ipynb"))
        for p in py_files
    ]

    if check:
        stale = check_files(pairs)
        if stale:
            print(
                f"\n✗ {stale} notebook(s) out of date. "
                f"Run: python convert_py_to_ipynb.py"
            )
            sys.exit(1)
        print(f"✓ {len(pairs)} notebooks up to date.")
        return

    print(f"Found {len(py_files)} files to convert\n")
    for py_file, output_file in pairs:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            convert_file(py_file, output_file)
        except Exception as e:
            print(f"  ✗ Error converting {py_file.name}: {e}")

    print(f"\n✓ Conversion complete! {len(py_files)} files processed.")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
