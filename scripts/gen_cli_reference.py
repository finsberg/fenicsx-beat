"""Regenerate docs/cli_reference.md from the beat.cli.config pydantic models.

Run after any change to src/beat/cli/config.py:

    python scripts/gen_cli_reference.py

tests/test_cli_reference.py fails if docs/cli_reference.md is out of date.
"""

from pathlib import Path

from beat.cli.config_reference import render_reference

out = Path(__file__).parents[1] / "docs" / "cli_reference.md"
out.write_text(render_reference())
print(f"Wrote {out}")
