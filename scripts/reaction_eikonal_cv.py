"""Conduction velocities of the monodomain R-D model used in the wedge benchmark.

TNNP 2006 (epi), Neic 2017 wedge conductivities (harmonic mean), on a 10 x 1.5 x 1.5 mm slab
stimulated at x <= 1 mm; CV from the threshold activation times at x = 4 and x = 8 mm on the
centre line, with the fibres along x (longitudinal) or y (transverse).

    mpirun -n 8 python scripts/reaction_eikonal_cv.py --dx 0.1 \
        --output scripts/reaction_eikonal_cv.json
"""

import argparse
import json
from pathlib import Path

from mpi4py import MPI

import dolfinx
import gotranx
import numpy as np
import scifem

import beat
from beat.activation import ActivationTracker

ODE = (
    Path(__file__).resolve().parents[1]
    / "odes"
    / "tentusscher_panfilov_2006"
    / "tentusscher_panfilov_2006_epi_cell.ode"
)


def cell_model():
    code = gotranx.cli.gotran2py.get_code(
        gotranx.load_ode(ODE),
        scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    )
    model: dict = {}
    exec(code, model)
    return model


def conduction_velocity(comm, dx, dt, direction):
    mesh = beat.geometry.get_3D_slab_mesh(comm, dx, 10.0, 1.5, 1.5)
    f = [1.0, 0.0, 0.0] if direction == "longitudinal" else [0.0, 1.0, 0.0]
    f0 = dolfinx.fem.Constant(mesh, np.array(f))
    g = beat.conductivities.default_conductivities("Neic2017_wedge")
    M = beat.conductivities.define_conductivity_tensor(f0=f0, **g)
    C_m = beat.units.ureg.Quantity(1.0, "uF/cm**2").to("uF/mm**2").magnitude
    time = dolfinx.fem.Constant(mesh, 0.0)
    cells = dolfinx.mesh.locate_entities(mesh, 3, lambda x: x[0] <= 1.0 + 1e-10)
    tags = dolfinx.mesh.meshtags(mesh, 3, cells, np.full(len(cells), 1, dtype=np.int32))
    I_s = beat.stimulation.define_stimulus(
        mesh=mesh,
        chi=g["chi"],
        time=time,
        subdomain_data=tags,
        marker=1,
        mesh_unit="mm",
        amplitude=50_000.0,
    )
    pde = beat.MonodomainModel(
        time=time,
        mesh=mesh,
        M=M,
        I_s=I_s,
        C_m=C_m,
        dx=I_s.dZ,
        params={
            "petsc_options": {"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg"},
        },
    )
    model = cell_model()
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    init = model["init_state_values"]()
    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=pde.state,
        fun=model["generalized_rush_larsen"],
        init_states=init,
        parameters=model["init_parameter_values"](stim_amplitude=0.0),
        num_states=len(init),
        v_index=model["state_index"]("V"),
    )
    solver = beat.SplittingSolver(pde=pde, ode=ode)
    tracker = ActivationTracker(pde.state, method="threshold")
    t = 0.0
    end = 25.0 if direction == "longitudinal" else 40.0
    while t < end - 1e-9:
        solver.step((t, t + dt))
        t += dt
        tracker.update(t)
    probes = np.array([[4.0, 0.75, 0.75], [8.0, 0.75, 0.75]])
    t4, t8 = scifem.evaluate_function(tracker.t_a, probes).ravel()
    return 4.0 / (t8 - t4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dx", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--output", type=Path, default=Path("reaction_eikonal_cv.json"))
    args = p.parse_args()
    comm = MPI.COMM_WORLD
    result = {"dx": args.dx, "dt": args.dt}
    for direction in ("longitudinal", "transverse"):
        result[direction] = conduction_velocity(comm, args.dx, args.dt, direction)
    if comm.rank == 0:
        print(json.dumps(result, indent=2))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
