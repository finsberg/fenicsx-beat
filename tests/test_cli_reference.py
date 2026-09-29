from pathlib import Path

from beat.cli.config_reference import render_reference


def test_reference_mentions_every_section_and_is_up_to_date():
    text = render_reference()
    for section in (
        "[geometry]",
        "[ep]",
        "[cell]",
        "[[stimulus]]",
        "[solver]",
        "[output]",
        "[postprocess]",
    ):
        assert section in text
    doc = Path(__file__).parents[1] / "docs" / "cli_reference.md"
    assert doc.read_text() == text, "Run `python scripts/gen_cli_reference.py`"
