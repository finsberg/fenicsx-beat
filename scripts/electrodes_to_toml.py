"""Convert a header-less electrode CSV to a ``[postprocess.ecg]`` TOML table.

The legacy simcardems / Alya format has one ``x,y,z`` row per electrode, in the order
LA, RA, LL, RL, V1 to V6. Run it once and append the output to the config:

    python scripts/electrodes_to_toml.py electrodes.csv --unit cm > ecg.toml
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ALYA_NAMES = "LA,RA,LL,RL,V1,V2,V3,V4,V5,V6"
BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def key(name: str) -> str:
    return name if BARE_KEY.match(name) else json.dumps(name)


def convert(path: Path, names: list[str], unit: str | None) -> str:
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: the electrode names are not unique: {names}")
    try:
        rows = np.loadtxt(path, delimiter=",", ndmin=2)
    except (OSError, ValueError) as e:
        raise ValueError(f"{path}: {e}") from e
    if rows.shape[1] not in (2, 3):
        raise ValueError(f"{path}: rows have {rows.shape[1]} columns, expected 2 or 3")
    if len(rows) != len(names):
        raise ValueError(f"{path}: {len(rows)} rows but {len(names)} names ({','.join(names)})")
    lines = ["[postprocess.ecg]"]
    if unit is not None:
        lines.append(f"unit = {json.dumps(unit)}")
    lines += ["", "[postprocess.ecg.electrodes]"]
    for name, row in zip(names, rows):
        lines.append(f"{key(name)} = [{', '.join(repr(float(x)) for x in row)}]")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("file", type=Path, help="header-less CSV, one x,y(,z) row per electrode")
    p.add_argument("--unit", help="length unit of the positions (default: geometry.unit)")
    p.add_argument("--names", default=ALYA_NAMES, help=f"comma-separated (default: {ALYA_NAMES})")
    p.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    args = p.parse_args(argv)
    try:
        text = convert(args.file, args.names.split(","), args.unit)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
