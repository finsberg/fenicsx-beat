"""Command line interface for fenicsx-beat (``beat``).

Requires the ``cli`` extra: ``pip install "fenicsx-beat[cli]"``.
"""

from typing import Optional, Sequence

from ._draft_cli import dispatch, setup_parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return dispatch(setup_parser(), argv)


__all__ = ["main"]
