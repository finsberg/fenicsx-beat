"""Verification of the bidomain model.

The manufactured pair below is built from Fourier modes whose normal derivative vanishes
on the unit square, so it satisfies the insulating boundary conditions of the model
exactly, and whose mean vanishes, so the extracellular potential meets the zero-mean
normalisation and the elliptic equation stays consistent.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import beat

# Intracellular and extracellular conductivities, as the diagonals of two tensors that are
# deliberately *not* proportional: an equal anisotropy ratio would make the extracellular
# potential a multiple of the transmembrane one and the coupling would go untested.
MI = (1.0, 0.4)
ME = (0.3, 1.2)

# Amplitude of the second spatial mode in v. Both modes are needed for the same reason.
GAMMA = 0.7


def M_i():
    return ufl.as_matrix([[MI[0], 0.0], [0.0, MI[1]]])


def M_e():
    return ufl.as_matrix([[ME[0], 0.0], [0.0, ME[1]]])


def mode(x, m, n):
    return ufl.cos(2 * ufl.pi * m * x[0]) * ufl.cos(2 * ufl.pi * n * x[1])


def _amplitudes():
    """Amplitudes of u_e that solve the elliptic equation exactly, mode by mode.

    Each mode is an eigenfunction of both conductivity operators, so the elliptic row
    decouples into one scalar equation per mode and no manufactured source is needed on
    that row.
    """
    a, b = MI
    c, d = ME
    return (
        -(a + b) / (a + b + c + d),
        -GAMMA * (4 * a + b) / (4 * (a + c) + (b + d)),
    )


def v_exact(x, t):
    return (mode(x, 1, 1) + GAMMA * mode(x, 2, 1)) * ufl.sin(t)


def ue_exact(x, t):
    beta_1, beta_2 = _amplitudes()
    return (beta_1 * mode(x, 1, 1) + beta_2 * mode(x, 2, 1)) * ufl.sin(t)


def stimulus(x, t, C_m=1.0):
    """The source that makes the manufactured pair solve the transmembrane equation."""
    v = v_exact(x, t)
    u_e = ue_exact(x, t)
    return C_m * ufl.diff(v, t) - ufl.div(M_i() * ufl.grad(v)) - ufl.div(M_i() * ufl.grad(u_e))


def l2(expr, mesh):
    form = dolfinx.fem.form(
        ufl.inner(expr, expr) * ufl.dx(domain=mesh, metadata={"quadrature_degree": 8}),
    )
    return np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM))


def mean(expr, mesh):
    dx = ufl.dx(domain=mesh, metadata={"quadrature_degree": 8})
    one = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))

    def integrate(integrand):
        return mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(integrand * dx)),
            op=MPI.SUM,
        )

    return integrate(expr) / integrate(one)


def spread(u):
    """max(u) - min(u) over all ranks, ignoring ghost entries."""
    local = u.x.array[: u.function_space.dofmap.index_map.size_local]
    comm = u.function_space.mesh.comm
    return comm.allreduce(local.max(), op=MPI.MAX) - comm.allreduce(local.min(), op=MPI.MIN)


def unit_square(N):
    return dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        N,
        N,
        dolfinx.cpp.mesh.CellType.triangle,
    )


def build(mesh, theta=0.5, C_m=1.0, **kwargs):
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    x = ufl.SpatialCoordinate(mesh)
    t = ufl.variable(time)
    return beat.BidomainModel(
        time=time,
        mesh=mesh,
        M_i=M_i(),
        M_e=M_e(),
        I_s=stimulus(x, t, C_m=C_m),
        C_m=C_m,
        params=dict(theta=theta),
        **kwargs,
    )


def test_manufactured_pair_solves_the_elliptic_equation():
    """Guard on the algebra behind the manufactured solution.

    If the pair did not satisfy the elliptic row exactly, the convergence tests below
    would be measuring against something the model is not being asked to solve.
    """
    mesh = unit_square(4)
    x = ufl.SpatialCoordinate(mesh)
    t = ufl.variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.7)))

    residual = ufl.div(M_i() * ufl.grad(v_exact(x, t))) + ufl.div(
        (M_i() + M_e()) * ufl.grad(ue_exact(x, t)),
    )

    assert l2(residual, mesh) < 1e-10


def test_manufactured_extracellular_potential_has_zero_mean():
    mesh = unit_square(4)
    x = ufl.SpatialCoordinate(mesh)
    t = ufl.variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.7)))

    assert abs(mean(ue_exact(x, t), mesh)) < 1e-12


@pytest.mark.parametrize("field", ["v", "u_e"])
# A capacitance away from one, so that dropping it anywhere in the transmembrane row
# changes the answer rather than cancelling.
@pytest.mark.parametrize("C_m", [1.0, 2.7])
def test_bidomain_spatial_convergence(field, C_m):
    dt = 0.001
    T = 10 * dt

    errors = []
    # Starting at N = 8: the second spatial mode fits two full wavelengths across the
    # domain, and a 4 x 4 mesh has not begun to resolve it.
    for N in (8, 16, 32, 64):
        mesh = unit_square(N)
        model = build(mesh, C_m=C_m)
        model.solve((0.0, T), dt=dt)

        x = ufl.SpatialCoordinate(mesh)
        exact = {"v": v_exact, "u_e": ue_exact}[field](x, T)
        errors.append(l2(getattr(model, field) - exact, mesh))

    rates = [np.log2(e1 / e2) for e1, e2 in zip(errors[:-1], errors[1:])]
    assert all(rate > 1.9 for rate in rates), (errors, rates)


@pytest.mark.parametrize("field", ["v", "u_e"])
def test_bidomain_temporal_convergence(field):
    """Measured against a finely stepped solve on the same mesh, not against the exact
    solution: on any mesh that a test can afford, the space discretization error is far
    larger than the time discretization error being measured and would swamp it."""
    T = 0.5
    mesh = unit_square(32)

    reference = build(mesh)
    reference.solve((0.0, T), dt=T / 2**9)
    reference_field = getattr(reference, field).copy()

    errors = []
    for dt in (T / 2**level for level in (2, 3, 4, 5)):
        model = build(mesh)
        model.solve((0.0, T), dt=dt)
        errors.append(l2(getattr(model, field) - reference_field, mesh))

    rates = [np.log2(e1 / e2) for e1, e2 in zip(errors[:-1], errors[1:])]
    assert all(rate > 1.8 for rate in rates), (errors, rates)


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_bidomain_reduces_to_monodomain_under_equal_anisotropy(theta):
    """With M_e = lambda * M_i the two models must agree step for step.

    The bidomain system then collapses algebraically onto the monodomain one with
    M = lambda / (1 + lambda) * M_i, and since u_e = -v / (1 + lambda) is representable in
    the same P1 space, the collapse is exact at the discrete level too.

    Both models start from a nonzero transmembrane potential, which is what an ODE solver
    would hand them. The collapse only survives that at theta < 1 if the model first
    brings its previous extracellular potential into agreement with it.
    """
    lam = 2.5
    dt = 0.01
    mesh = unit_square(16)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    x = ufl.SpatialCoordinate(mesh)
    t = ufl.variable(time)
    I_s = stimulus(x, t)

    bidomain = beat.BidomainModel(
        time=time,
        mesh=mesh,
        M_i=M_i(),
        M_e=lam * M_i(),
        I_s=I_s,
        params=dict(theta=theta),
    )
    monodomain = beat.MonodomainModel(
        time=time,
        mesh=mesh,
        M=(lam / (1.0 + lam)) * M_i(),
        I_s=I_s,
        params=dict(theta=theta),
    )

    def initial_v(x):
        return 0.3 * np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + 0.1

    bidomain.v_.interpolate(initial_v)
    monodomain.v_.interpolate(initial_v)

    for step in range(8):
        interval = (step * dt, (step + 1) * dt)
        bidomain.step(interval)
        monodomain.step(interval)

        difference = l2(bidomain.v - monodomain.v, mesh)
        scale = l2(monodomain.v, mesh)
        assert difference < 1e-11 * scale, (step, difference, scale)

        # u_e must track the reduction that makes the collapse exact. Only up to a
        # constant: the reduction fixes the field's shape, the zero-mean normalisation
        # fixes its datum, and the two need not agree.
        residual = bidomain.u_e + bidomain.v / (1.0 + lam)
        offset = mean(residual, mesh)
        assert l2(residual - offset, mesh) < 1e-11 * scale, (step, offset)

        bidomain.assign_previous()
        monodomain.assign_previous()


def test_grounding_does_not_change_the_solution():
    """v and the spread of u_e are physical; the datum of u_e is not.

    The ground is a single degree of freedom rather than a whole boundary. Holding an
    entire edge at one value is an extra constraint, not a choice of datum: the exact u_e
    is not constant along any edge of this domain.
    """
    dt = 0.005
    T = 10 * dt

    def ground_a_corner_at(value):
        def factory(model):
            dofs = dolfinx.fem.locate_dofs_geometrical(
                model.V_ue,
                lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
            )
            return [
                dolfinx.fem.dirichletbc(
                    dolfinx.default_scalar_type(value),
                    dofs,
                    model.V_ue,
                ),
            ]

        return factory

    results = {}
    for name, bcs in (
        ("zero mean", None),
        ("grounded at 0", ground_a_corner_at(0.0)),
        ("grounded at 7", ground_a_corner_at(7.0)),
    ):
        mesh = unit_square(16)
        model = build(mesh, bcs=bcs)
        model.solve((0.0, T), dt=dt)
        x = ufl.SpatialCoordinate(mesh)
        results[name] = (
            l2(model.v - v_exact(x, T), mesh),
            spread(model.u_e),
            model.u_e.x.array.mean(),
        )

    reference = results["zero mean"]
    for name, (v_error, ue_spread, _) in results.items():
        assert np.isclose(v_error, reference[0], rtol=1e-8), (name, results)
        assert np.isclose(ue_spread, reference[1], rtol=1e-8), (name, results)

    # ...and the datum really did move, so the check above is not comparing three
    # identical solves.
    assert results["grounded at 7"][2] > results["grounded at 0"][2] + 1.0


def simple_ode_forward_euler(states, t, dt, parameters):
    v, s = states
    values = np.zeros_like(states)
    values[0] = v - s * dt
    values[1] = s + v * dt
    return values


def test_splitting_solver_drives_the_bidomain_model():
    """The splitting solver only ever asks the PDE model to step and to keep history, so
    the bidomain model must drop into it in place of the monodomain one -- and under equal
    anisotropy the two must then produce the same transmembrane potential."""
    lam = 2.5
    dt = 0.01
    T = 20 * dt
    mesh = unit_square(16)
    x = ufl.SpatialCoordinate(mesh)

    def solve_with(pde):
        V_ode = beat.utils.space_from_string("P_1", mesh, dim=1)
        v_ode = dolfinx.fem.Function(V_ode)
        init_states = np.zeros((2, v_ode.x.array.size))
        init_states[1, :] = 0.2
        ode = beat.odesolver.DolfinODESolver(
            v_ode=v_ode,
            v_pde=pde.v,
            fun=simple_ode_forward_euler,
            init_states=init_states,
            parameters=None,
            num_states=2,
            v_index=0,
        )
        beat.MonodomainSplittingSolver(pde=pde, ode=ode).solve((0.0, T), dt=dt)
        return pde.v

    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    t = ufl.variable(time)
    I_s = stimulus(x, t)

    v_bidomain = solve_with(
        beat.BidomainModel(time=time, mesh=mesh, M_i=M_i(), M_e=lam * M_i(), I_s=I_s),
    )
    time.value = 0.0
    v_monodomain = solve_with(
        beat.MonodomainModel(time=time, mesh=mesh, M=(lam / (1.0 + lam)) * M_i(), I_s=I_s),
    )

    scale = l2(v_monodomain, mesh)
    assert scale > 0.05, scale
    assert l2(v_bidomain - v_monodomain, mesh) < 1e-11 * scale


def test_bidomain_state_is_both_fields():
    mesh = unit_square(4)
    model = build(mesh)

    assert tuple(model.state) == (model.v, model.u_e)
    assert model.state.v is model.v
    assert model.state.u_e is model.u_e
