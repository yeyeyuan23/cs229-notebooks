#!/usr/bin/env python3
"""Validate and execute all CS229 practice notebooks without writing outputs."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SECTIONS = ("## Goal", "## Setup", "## Steps", "## Checks", "## Next Steps")


def validate_one(path: Path) -> tuple[int, int]:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)

    lecture_match = re.search(r"lecture(\d{2})", path.name)
    if not lecture_match:
        raise AssertionError(f"missing lecture number in filename: {path}")
    lecture_number = int(lecture_match.group(1))

    metadata = notebook.metadata.get("cs229", {})
    if metadata.get("offering") != "Spring 2026":
        raise AssertionError(f"wrong offering metadata: {path}")
    if metadata.get("lecture") != lecture_number:
        raise AssertionError(f"lecture metadata mismatch: {path}")

    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    for section in REQUIRED_SECTIONS:
        if section not in markdown:
            raise AssertionError(f"missing {section}: {path}")
    if "https://www.youtube.com/watch?v=" not in markdown:
        raise AssertionError(f"missing official video URL: {path}")
    if "https://cs229.stanford.edu/main_notes.pdf" not in markdown:
        raise AssertionError(f"missing main notes URL: {path}")

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells:
        raise AssertionError(f"no code cells: {path}")
    if any("TODO" not in cell.source for cell in code_cells):
        raise AssertionError(f"non-TODO starter code cell: {path}")
    if any(cell.get("outputs") for cell in code_cells):
        raise AssertionError(f"starter notebook contains output: {path}")

    # Execute an in-memory copy. TODO cells contain comments only, so starter notebooks
    # must be safe to run top-to-bottom before the student fills them in.
    NotebookClient(notebook, timeout=60, kernel_name="python3").execute()
    return len(notebook.cells), len(code_cells)


def main() -> None:
    paths = sorted(ROOT.glob("lecture*/*.ipynb"))
    if len(paths) != 17:
        raise AssertionError(f"expected 17 notebooks, found {len(paths)}")

    observed_lectures = []
    for path in paths:
        cells, code_cells = validate_one(path)
        lecture_number = int(re.search(r"lecture(\d{2})", path.name).group(1))
        observed_lectures.append(lecture_number)
        print(f"PASS lecture {lecture_number:02d}: cells={cells}, code={code_cells}")

    if observed_lectures != list(range(1, 18)):
        raise AssertionError(f"lecture sequence mismatch: {observed_lectures}")
    print("PASS all 17 notebooks: schema, structure, sources, TODO mode, and in-memory execution")


if __name__ == "__main__":
    main()
