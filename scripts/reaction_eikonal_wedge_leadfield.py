"""Lead field of the wedge electrode P_E in its bath (phase 1 of the Fig. 7 pipeline).

    mpirun -n 16 python scripts/reaction_eikonal_wedge_leadfield.py --dx 0.1

Writes <outdir>/leadfield_dx<dx>.npy on rank 0: Z at the wedge's tissue grid vertices, read by
reaction_eikonal_wedge.py to record the electrogram phi_e(P_E, t).
"""

import argparse
import time as walltime
from pathlib import Path

from mpi4py import MPI

import numpy as np
from reaction_eikonal_wedge_grid import ELECTRODE, MARGIN, gather_to_grid, volume_conductor


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dx", type=float, default=0.4)
    p.add_argument("--margin", type=float, default=MARGIN)
    p.add_argument("--outdir", type=Path, default=Path("results-reaction-eikonal-bath"))
    args = p.parse_args()
    comm = MPI.COMM_WORLD
    start = walltime.perf_counter()
    bath, conductor = volume_conductor(comm, args.dx, args.margin)
    Z = conductor.lead_field(ELECTRODE)
    array = gather_to_grid(Z, args.dx)
    iterations = conductor.ksp.getIterationNumber()
    num_vertices = bath.mesh.topology.index_map(0).size_global
    if comm.rank == 0:
        if np.isnan(array).any():
            raise RuntimeError("The lead field is missing at some tissue grid vertices")
        args.outdir.mkdir(parents=True, exist_ok=True)
        path = args.outdir / f"leadfield_dx{args.dx:g}.npy"
        np.save(path, array)
        print(
            f"wrote {path}: {num_vertices} vertices, margins {bath.margins} mm, {iterations} CG "
            f"iterations, {walltime.perf_counter() - start:.0f} s",
        )


if __name__ == "__main__":
    main()
