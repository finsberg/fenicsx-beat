"""docs/architecture.md names every public class; docs/api.rst documents every module."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "beat"


def public_modules():
    return sorted(p for p in SRC.glob("*.py") if not p.stem.startswith("_"))


def public_classes():
    for path in public_modules():
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                yield path.stem, node.name


def test_scanner_finds_the_package():
    assert ("monodomain_solver", "SplittingSolver") in set(public_classes())
    assert "eikonal" in {p.stem for p in public_modules()}


def test_every_public_class_is_on_the_code_structure_page():
    text = (ROOT / "docs" / "architecture.md").read_text()
    missing = [f"{m}.{c}" for m, c in public_classes() if f"`{c}`" not in text]
    assert not missing, f"Add these classes to docs/architecture.md: {missing}"


def test_every_module_is_in_the_api_docs():
    text = (ROOT / "docs" / "api.rst").read_text()
    missing = [
        p.stem
        for p in public_modules()
        if not re.search(rf"^\.\. automodule:: beat\.{p.stem}$", text, flags=re.MULTILINE)
    ]
    assert not missing, f"Add these modules to docs/api.rst: {missing}"
