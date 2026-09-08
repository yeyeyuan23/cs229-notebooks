"""Validate and smoke-test the generated CS229 starter notebooks."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    ROOT / "problem_sets/ps0_foundations/cs229_simplified_ps0_foundations.ipynb",
    ROOT / "problem_sets/ps1_regression/cs229_simplified_ps1_regression.ipynb",
    ROOT / "problem_sets/ps2_classification/cs229_simplified_ps2_classification.ipynb",
    ROOT / "problem_sets/ps3_unsupervised/cs229_simplified_ps3_unsupervised.ipynb",
]
REQUIRED_HEADINGS = ("## Goal", "## Setup", "## Checks", "## Next Steps")


def validate_one(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    markdown = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")

    for heading in REQUIRED_HEADINGS:
        if heading not in markdown:
            raise AssertionError(f"{path}: missing {heading}")
    if "START TODO" not in code:
        raise AssertionError(f"{path}: no implementation TODO found")
    if nb.metadata.get("cs229", {}).get("offering") != "Spring 2026":
        raise AssertionError(f"{path}: wrong offering metadata")
    if nb.metadata.get("cs229", {}).get("artifact") != "simplified-source-grounded-problem-set":
        raise AssertionError(f"{path}: wrong artifact metadata")

    for cell in nb.cells:
        if cell.cell_type == "code":
            if cell.get("execution_count") is not None or cell.get("outputs"):
                raise AssertionError(f"{path}: starter contains saved execution state")

    with tempfile.TemporaryDirectory(prefix="cs229-nb-") as runtime_dir:
        os.environ["JUPYTER_RUNTIME_DIR"] = runtime_dir
        client = NotebookClient(
            nb,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()


def main() -> int:
    missing = [path for path in EXPECTED if not path.exists()]
    if missing:
        for path in missing:
            print(f"MISSING {path.relative_to(ROOT)}")
        return 1

    for path in EXPECTED:
        validate_one(path)
        print(f"PASS {path.relative_to(ROOT)}")
    print(f"Validated {len(EXPECTED)} starter notebooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
