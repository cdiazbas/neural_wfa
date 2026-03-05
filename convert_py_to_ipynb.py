#!/usr/bin/env python
"""
Convert Python scripts from example_py/ to Jupyter notebooks in examples_ipynb/

This script parses Python files that use the special comment-based notebook format
(with '# In[ ]:' markers and markdown comments) and converts them to proper .ipynb files.

Usage:
    python convert_py_to_ipynb.py [file1.py file2.py ...] 
    
    If no files specified, converts all .py files in example_py/
"""

import sys
import os
import json
import re
from pathlib import Path


def parse_py_to_cells(py_content):
    """
    Parse a Python file with notebook-style comments into cells.
    
    Expected format:
    - Lines starting with '# #' are markdown headers
    - Lines starting with '# In[ ]:' start new code cells
    - Other lines starting with '#' followed by text are markdown
    - Other lines are code
    
    Returns:
        list: List of dicts with 'cell_type' and 'source' keys
    """
    lines = py_content.split('\n')
    cells = []
    current_cell = None
    current_source = []
    
    i = 0
    # Skip shebang and encoding lines
    while i < len(lines) and (lines[i].startswith('#!') or 
                               lines[i].startswith('# coding:') or 
                               lines[i].startswith('# -*- coding:')):
        i += 1
    
    # Skip empty lines at the start
    while i < len(lines) and lines[i].strip() == '':
        i += 1
    
    def save_current_cell():
        """Helper to save the current cell being built"""
        if current_cell and current_source:
            # Clean up trailing empty lines
            while current_source and current_source[-1].strip() == '':
                current_source.pop()
            
            if current_cell == 'markdown':
                # Remove leading '#' and space from markdown lines
                cleaned = []
                for line in current_source:
                    if line.startswith('# '):
                        cleaned.append(line[2:])
                    elif line.startswith('#'):
                        cleaned.append(line[1:])
                    else:
                        cleaned.append(line)
                cells.append({
                    'cell_type': 'markdown',
                    'source': cleaned
                })
            else:  # code
                cells.append({
                    'cell_type': 'code',
                    'source': current_source
                })
    
    while i < len(lines):
        line = lines[i]
        
        # Check for cell marker
        if line.strip() == '# In[ ]:' or line.strip().startswith('# In['):
            # Save previous cell
            save_current_cell()
            current_cell = 'code'
            current_source = []
            i += 1
            continue
        
        # Check for markdown content (lines starting with '# ')
        if line.startswith('# ') or (line.startswith('#') and not line.startswith('#!/')):
            # If we were in a code cell, save it
            if current_cell == 'code':
                save_current_cell()
                current_cell = 'markdown'
                current_source = []
            elif current_cell is None:
                current_cell = 'markdown'
                current_source = []
            
            current_source.append(line)
        elif line.strip() == '' and current_cell == 'markdown':
            # Empty line in markdown
            current_source.append(line)
        else:
            # Code line
            if current_cell == 'markdown':
                save_current_cell()
                current_cell = 'code'
                current_source = []
            elif current_cell is None:
                current_cell = 'code'
                current_source = []
            
            current_source.append(line)
        
        i += 1
    
    # Save the last cell
    save_current_cell()
    
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
    # Get the script directory
    script_dir = Path(__file__).parent
    input_dir = script_dir / 'example_py'
    output_dir = script_dir / 'examples_ipynb'
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Get files to convert
    if len(sys.argv) > 1:
        # Convert specific files
        py_files = [Path(f) for f in sys.argv[1:]]
    else:
        # Convert all .py files in example_py/
        py_files = sorted(input_dir.glob('*.py'))
    
    if not py_files:
        print("No Python files found to convert.")
        return
    
    print(f"Found {len(py_files)} files to convert\n")
    
    # Convert each file
    for py_file in py_files:
        if not py_file.exists():
            print(f"Warning: {py_file} does not exist, skipping")
            continue
        
        # Create output path
        output_file = output_dir / py_file.with_suffix('.ipynb').name
        
        try:
            convert_file(py_file, output_file)
        except Exception as e:
            print(f"  ✗ Error converting {py_file.name}: {e}")
    
    print(f"\n✓ Conversion complete! {len(py_files)} files processed.")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
