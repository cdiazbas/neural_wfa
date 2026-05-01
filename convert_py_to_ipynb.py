#!/usr/bin/env python
"""
Convert Python scripts to Jupyter notebooks.

The canonical format uses VS Code / Spyder cell markers:
    # %%              → code cell boundary
    # %% [markdown]   → markdown cell boundary
                        (content follows as comment lines: # text)

Usage:
    # Batch: convert all .py files in examples_py/ → examples_ipynb/
    python convert_py_to_ipynb.py

    # Single file
    python convert_py_to_ipynb.py <input.py> <output.ipynb>
"""

import sys
import json
from pathlib import Path


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
        # Remove trailing blank lines
        while current_source and current_source[-1].strip() == '':
            current_source.pop()
        if not current_source:
            return
        if current_type == 'markdown':
            cleaned = []
            for line in current_source:
                if line.startswith('# '):
                    cleaned.append(line[2:])
                elif line.startswith('#'):
                    cleaned.append(line[1:])
                else:
                    cleaned.append(line)
            cells.append({'cell_type': 'markdown', 'source': cleaned})
        else:
            cells.append({'cell_type': 'code', 'source': list(current_source)})

    for line in lines:
        stripped = line.strip()
        if stripped == '# %%' or stripped.startswith('# %% ') and '[markdown]' not in stripped:
            flush()
            current_type = 'code'
            current_source = []
        elif stripped.startswith('# %%') and '[markdown]' in stripped:
            flush()
            current_type = 'markdown'
            current_source = []
        else:
            if current_type is None:
                # Content before any marker → code cell
                current_type = 'code'
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
        if cell['cell_type'] == 'markdown':
            nb_cell = {
                'cell_type': 'markdown',
                'metadata': {},
                'source': cell['source']
            }
        else:  # code
            nb_cell = {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'source': cell['source'],
                'outputs': []
            }
        nb_cells.append(nb_cell)
    
    notebook = {
        'cells': nb_cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {
                    'name': 'ipython',
                    'version': 3
                },
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.9.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    return notebook


def convert_file(input_path, output_path):
    """
    Convert a single Python file to a Jupyter notebook.
    
    Args:
        input_path: Path to input .py file
        output_path: Path to output .ipynb file
    """
    print(f"Converting {input_path.name} -> {output_path.name}")
    
    # Read the Python file
    with open(input_path, 'r', encoding='utf-8') as f:
        py_content = f.read()
    
    # Parse into cells
    cells = parse_py_to_cells(py_content)
    
    # Create notebook structure
    notebook = create_notebook(cells)
    
    # Write the notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"  ✓ Created {output_path.name} with {len(cells)} cells")


def main():
    """Main conversion function"""
    script_dir = Path(__file__).parent

    # Single-file mode: convert_py_to_ipynb.py <input.py> <output.ipynb>
    if len(sys.argv) == 3:
        input_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        convert_file(input_path, output_path)
        return

    # Batch mode: convert all .py files in examples_py/ → examples_ipynb/
    input_dir = script_dir / 'examples_py'
    output_dir = script_dir / 'examples_ipynb'
    output_dir.mkdir(exist_ok=True)

    py_files = sorted(input_dir.glob('*.py'))
    if not py_files:
        print("No Python files found to convert.")
        return

    print(f"Found {len(py_files)} files to convert\n")
    for py_file in py_files:
        output_file = output_dir / py_file.with_suffix('.ipynb').name
        try:
            convert_file(py_file, output_file)
        except Exception as e:
            print(f"  ✗ Error converting {py_file.name}: {e}")

    print(f"\n✓ Conversion complete! {len(py_files)} files processed.")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
