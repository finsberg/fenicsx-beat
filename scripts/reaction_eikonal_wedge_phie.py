"""phi_e maps of the Fig. 7 snapshots in the wedge's bath (phase 3 of the Fig. 7 pipeline).

    mpirun -n 16 python scripts/reaction_eikonal_wedge_phie.py --model rd --dx 0.1

Reads <outdir>/<model>_dx<dx>_v_t<t>.npy (written by reaction_eikonal_wedge.py), solves the
pseudo-bidomain volume conductor for each and writes <outdir>/phie_<model>_dx<dx>.npz on rank 0
(phi_e on the 101 x 101 mid-plane grid, mV).
"""

import argparse
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
from reaction_eikonal_wedge_grid import (
    MARGIN,
    SNAPSHOTS,
    midplane_grid,
    scatter_from_grid,
    volume_conductor,
)

from beat.utils import PointEvaluator


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", choices=["rd", "re+", "re-"], required=True)
    p.add_argument("--dx", type=float, default=0.4)
    p.add_argument("--margin", type=float, default=MARGIN)
    p.add_argument("--outdir", type=Path, default=Path("results-reaction-eikonal-bath"))
    args = p.parse_args()
    comm = MPI.COMM_WORLD
    name = f"{args.model}_dx{args.dx:g}"
    snapshots = {t: args.outdir / f"{name}_v_t{t:g}.npy" for t in SNAPSHOTS}
    snapshots = {t: path for t, path in snapshots.items() if path.is_file()}
    if not snapshots:
        raise SystemExit(f"no snapshots {name}_v_t*.npy in {args.outdir}")
    bath, conductor = volume_conductor(comm, args.dx, args.margin)
    gx, gy, points = midplane_grid()
    evaluate = PointEvaluator(bath.mesh, points)
    v = dolfinx.fem.Function(conductor.V, name="v")
    maps = {}
    for t, path in snapshots.items():
        scatter_from_grid(v, np.load(path), args.dx)
        maps[f"phie_t{t:g}"] = evaluate(conductor.solve(v)).reshape(gx.shape)
    if comm.rank == 0:
        np.savez(
            args.outdir / f"phie_{name}.npz",
            grid_x=gx,
            grid_y=gy,
            margins=bath.margins,
            **maps,
        )
        print(f"wrote {args.outdir / f'phie_{name}.npz'} ({', '.join(maps)})")


if __name__ == "__main__":
    main()
