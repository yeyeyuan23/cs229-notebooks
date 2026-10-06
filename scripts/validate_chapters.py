#!/usr/bin/env python3
"""Validate the active CS229 chapter or a fresh blank template without rewriting it."""
from __future__ import annotations
import argparse
import ast
import os
from pathlib import Path
import tempfile
import nbformat
from nbclient import NotebookClient
from build_chapters import build_chapter, NOTEBOOK_PATH, ROOT


def validate_structure(nb):
    nbformat.validate(nb)
    markdown = '\n'.join(c.source for c in nb.cells if c.cell_type == 'markdown')
    code = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    for heading in ['## Goal', '## Setup', '## 回归的准备', '## 1.1', '## 1.2', '## 同一个回归实验', '## 1.3', '## 1.4', '## Checks', '## Next Steps']:
        assert heading in markdown, f'Missing {heading}'
    assert nb.metadata.cs229.learning_mode == 'notes-to-code'
    assert nb.metadata.cs229.notes_date == '2026-08-23'
    assert '你的答案（请双击本格后填写）' not in markdown
    assert 'RUN_P' not in code and 'TRY_' not in code
    # Catch accidental four-space indentation turning ordinary prose into code blocks.
    for c in nb.cells:
        if c.cell_type == 'markdown':
            assert not any(line.startswith('    ') for line in c.source.splitlines()), 'Indented Markdown prose'
        else:
            ast.parse(c.source)
    for function in ['prepare_housing', 'predict_loop', 'squared_error_gradient', 'fit_sgd', 'torch_gradient', 'fit_torch_gradient_descent', 'compare_numpy_torch', 'gaussian_log_likelihood', 'compare_models']:
        assert f'def {function}(' in code


def execute(nb, kernel_name='python3'):
    with tempfile.TemporaryDirectory(prefix='cs229-chapter-') as runtime:
        os.environ['JUPYTER_RUNTIME_DIR'] = runtime
        os.environ['MPLCONFIGDIR'] = runtime
        os.environ['IPYTHONDIR'] = str(Path(runtime) / 'ipython')
        os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
        NotebookClient(nb, timeout=120, kernel_name=kernel_name, resources={'metadata': {'path': str(ROOT)}}).execute()
    pending = []
    for c in nb.cells:
        for output in c.get('outputs', []):
            if output.output_type == 'stream':
                pending.extend(line for line in output.text.splitlines() if line.startswith('待完成：'))
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starters', action='store_true', help='check the blank template in memory')
    parser.add_argument('--kernel', default='python3', help='Jupyter kernel used for execution')
    parser.add_argument('--require-torch', action='store_true', help='fail when PyTorch is unavailable instead of skipping its exercises')
    args = parser.parse_args()
    nb = build_chapter() if args.starters else nbformat.read(NOTEBOOK_PATH, as_version=4)
    validate_structure(nb)
    pending = execute(nb, kernel_name=args.kernel)
    missing = [line for c in nb.cells for o in c.get('outputs', []) if o.output_type == 'stream'
               for line in o.text.splitlines() if line.startswith('缺少依赖：')]
    if args.require_torch and missing:
        raise RuntimeError('The selected kernel lacks PyTorch; install it in that kernel environment')
    print('PASS: notebook structure and top-to-bottom execution')
    print(f'{len(pending)} exercises await learner implementation; this is not a claim that they are solved.')
    for line in missing:
        print(line)
    if missing:
        print('PyTorch exercises were skipped, not validated. Use --require-torch after installing torch.')


if __name__ == '__main__':
    main()
