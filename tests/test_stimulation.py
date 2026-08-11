from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import beat
import beat.units


def test_single_stimulation():
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_interval(comm, 10)
    value = 2.0
    end = 1.0
    start = 0.5
    dt = 0.01
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))

    expr = ufl.conditional(ufl.And(ufl.ge(time, start), ufl.le(time, end)), value, 0.0)
    I_s = beat.stimulation.Stimulus(dZ=ufl.dx(domain=mesh), expr=expr)

    pde = beat.MonodomainModel(
        time=time,
        mesh=mesh,
        M=dolfinx.fem.Constant(mesh, 0.0),
        I_s=I_s,
    )

    pde.step((0.0, 0.4))
    assert np.allclose(pde.state.x.array, 0.0)

    t0 = 0.9
    stim_t0 = value * (t0 - start)
    pde.solve((0.4, t0), dt=dt)

    # At time dt the stimulus should be value and since M is zero the state should be value * dt
    assert np.allclose(pde.state.x.array, stim_t0)

    pde.solve((t0, end + dt), dt=dt)

    # At end the stimulus should be zero and since M is zero the state should be zero
    assert np.allclose(pde.state.x.array, (end - start - dt) * value)

    # Solving for longer time should not change the state
    pde.solve((end + dt, 2 * end), dt=dt)
    assert np.allclose(pde.state.x.array, (end - start - dt) * value)


def test_double_stimulation():
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_interval(comm, 10)
    dt = 0.01
    value1 = 2.0
    value2 = 3.0
    start1 = 0.5
    end1 = 1.0
    start2 = 0.9
    end2 = 1.5

    time = dolfinx.fem.Constant(mesh, 0.0)
    expr1 = ufl.conditional(ufl.And(ufl.ge(time, start1), ufl.le(time, end1)), value1, 0.0)
    expr2 = ufl.conditional(ufl.And(ufl.ge(time, start2), ufl.le(time, end2)), value2, 0.0)
    dx = ufl.dx(domain=mesh)
    I_s = [
        beat.stimulation.Stimulus(dZ=dx, expr=expr1),
        beat.stimulation.Stimulus(dZ=dx, expr=expr2),
    ]

    pde = beat.MonodomainModel(
        time=time,
        mesh=mesh,
        M=dolfinx.fem.Constant(mesh, 0.0),
        I_s=I_s,
    )

    pde.step((0.0, 0.4))
    assert np.allclose(pde.state.x.array, 0.0)

    # Solve up to the second stimulus starts
    t0 = 0.9
    stim_t0 = value1 * (t0 - start1)
    pde.solve((0.4, t0), dt=dt)
    # At time dt the stimulus should be value and since M is zero the state should be value * dt
    assert np.allclose(pde.state.x.array, stim_t0)

    # Solve up to the end of the first stimulus
    pde.solve((t0, end1 + dt), dt=dt)
    assert np.allclose(
        pde.state.x.array,
        (end1 - start1 - dt) * value1 + (end1 + dt - start2) * value2,
    )

    # Solve up to the end of the second stimulus
    pde.solve((end1 + dt, end2 + dt), dt=dt)
    assert np.allclose(
        pde.state.x.array,
        (end1 - start1 - dt) * value1 + (end2 - start2 - dt) * value2,
    )

    # Solving for longer time should not change the state
    pde.solve((end2 + dt, 2 * end2), dt=dt)
    assert np.allclose(
        pde.state.x.array,
        (end1 - start1 - dt) * value1 + (end2 - start2 - dt) * value2,
    )


@pytest.mark.parametrize("subdomain_dim", [0, 1, 2, 3])
def test_effective_dim_3D(subdomain_dim):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 2, 2, 2, dolfinx.cpp.mesh.CellType.tetrahedron)
    entities = dolfinx.mesh.locate_entities(mesh, subdomain_dim, lambda x: np.full(x.shape[1], 1))
    subdomain_data = dolfinx.mesh.meshtags(
        mesh,
        subdomain_dim,
        entities,
        np.full(entities.shape, 1),
    )
    effective_dim = beat.stimulation.compute_effective_dim(mesh, subdomain_data)

    assert effective_dim == subdomain_dim


@pytest.mark.parametrize("subdomain_dim", [0, 1, 2])
def test_effective_dim_2D(subdomain_dim):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 2, 2, dolfinx.cpp.mesh.CellType.triangle)
    entities = dolfinx.mesh.locate_entities(mesh, subdomain_dim, lambda x: np.full(x.shape[1], 1))
    subdomain_data = dolfinx.mesh.meshtags(
        mesh,
        subdomain_dim,
        entities,
        np.full(entities.shape, 1),
    )
    effective_dim = beat.stimulation.compute_effective_dim(mesh, subdomain_data)

    assert effective_dim == subdomain_dim + 1


@pytest.mark.parametrize("subdomain_dim", [0, 1])
def test_effective_dim_1D(subdomain_dim):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_interval(comm, 2)
    entities = dolfinx.mesh.locate_entities(mesh, subdomain_dim, lambda x: np.full(x.shape[1], 1))
    subdomain_data = dolfinx.mesh.meshtags(
        mesh,
        subdomain_dim,
        entities,
        np.full(entities.shape, 1),
    )
    effective_dim = beat.stimulation.compute_effective_dim(mesh, subdomain_data)

    assert effective_dim == subdomain_dim + 2


@pytest.mark.parametrize("subdomain_dim, integral_type", [(1, "exterior_facet"), (2, "cell")])
def test_get_dZ_2D(subdomain_dim, integral_type):
    """
    Test the get_dZ function with different mesh and subdomain data.
    """
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 2, 2, dolfinx.cpp.mesh.CellType.triangle)
    cells = dolfinx.mesh.locate_entities(
        mesh,
        subdomain_dim,
        lambda x: np.logical_and(x[0] <= 0.5, x[1] <= 0.5),
    )
    stim_marker = 1
    stim_tags = dolfinx.mesh.meshtags(
        mesh,
        subdomain_dim,
        cells,
        np.full(len(cells), stim_marker, dtype=np.int32),
    )

    dZ = beat.stimulation.get_dZ(mesh, stim_tags)
    assert isinstance(dZ, ufl.Measure)
    assert dZ.integral_type() == integral_type


@pytest.mark.parametrize("subdomain_dim, integral_type", [(2, "exterior_facet"), (3, "cell")])
def test_get_dZ_3D(subdomain_dim, integral_type):
    """
    Test the get_dZ function with different mesh and subdomain data.
    """
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 2, 2, 2, dolfinx.cpp.mesh.CellType.tetrahedron)
    cells = dolfinx.mesh.locate_entities(
        mesh,
        subdomain_dim,
        lambda x: np.logical_and(x[0] <= 0.5, x[1] <= 0.5),
    )
    stim_marker = 1
    stim_tags = dolfinx.mesh.meshtags(
        mesh,
        subdomain_dim,
        cells,
        np.full(len(cells), stim_marker, dtype=np.int32),
    )

    dZ = beat.stimulation.get_dZ(mesh, stim_tags)
    assert isinstance(dZ, ufl.Measure)
    assert dZ.integral_type() == integral_type


@pytest.mark.parametrize(
    "effective_dim, mesh_unit, expected_unit",
    [
        (0, "cm", "uA"),
        (1, "cm", "uA"),
        (2, "cm", "uA/cm"),
        (3, "cm", "uA/cm**2"),
        (0, "m", "uA"),
        (1, "m", "uA"),
        (2, "m", "uA/m"),
    ],
)
def test_compute_stimulus_unit(effective_dim, mesh_unit, expected_unit):
    """
    Test the compute_stimulus_unit function with different effective
    dimensions and mesh units.
    """
    assert beat.stimulation.compute_stimulus_unit(effective_dim, mesh_unit) == beat.units.ureg(
        expected_unit,
    )


@pytest.mark.parametrize(
    "value, mesh_unit, expected_value",
    [
        (1.0, "cm", 1.0 * beat.units.ureg("cm**-1")),
        (2.0 * beat.units.ureg("mm**-1"), "cm", 2.0 * beat.units.ureg("mm**-1")),
    ],
)
def test_convert_chi(value, mesh_unit, expected_value):
    assert beat.stimulation.convert_chi(value, mesh_unit) == expected_value


@pytest.mark.parametrize(
    "effective_dim, amplitude, expected_value",
    [
        (1, 2.0, 2.0 * beat.units.ureg("uA / cm")),
        (2, 2.0, 2.0 * beat.units.ureg("uA / cm**2")),
        (3, 2.0, 2.0 * beat.units.ureg("uA / cm**3")),
    ],
)
def test_convert_amplitude(effective_dim, amplitude, expected_value):
    assert beat.stimulation.convert_amplitude(effective_dim, amplitude) == expected_value


def test_define_stimulus():
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 2, 2, dolfinx.cpp.mesh.CellType.triangle)
    # Create cells where all points are inside the subdomain
    cells = dolfinx.mesh.locate_entities(
        mesh,
        mesh.topology.dim,
        lambda x: np.full(x.shape[1], True),
    )
    stim_marker = 1
    stim_tags = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim,
        cells,
        np.full(len(cells), stim_marker, dtype=np.int32),
    )

    time = dolfinx.fem.Constant(mesh, 0.0)
    start = 1.0
    duration = 2.0
    amplitude = 3.0
    chi = 2.0
    stimulus = beat.stimulation.define_stimulus(
        mesh=mesh,
        chi=chi,
        time=time,
        amplitude=amplitude,
        start=start,
        duration=duration,
        mesh_unit="cm",
        marker=stim_marker,
        subdomain_data=stim_tags,
    )
    assert stimulus.marker == stim_marker
    stim_form = dolfinx.fem.form(stimulus.expr * stimulus.dz)
    # Stimulus should be zero at the start
    assert np.isclose(comm.allreduce(dolfinx.fem.assemble_scalar(stim_form), op=MPI.SUM), 0.0)
    # Stimulus should be non-zero at the start of stimulus
    time.value = start
    assert np.isclose(
        comm.allreduce(dolfinx.fem.assemble_scalar(stim_form), op=MPI.SUM),
        amplitude / chi,
    )
    # Stimulus should still be non-zero
    time.value = start + duration / 2
    assert np.isclose(
        comm.allreduce(dolfinx.fem.assemble_scalar(stim_form), op=MPI.SUM),
        amplitude / chi,
    )
    # Stimulus should be zero after the duration
    time.value = start + duration + 1e-6
    assert np.isclose(comm.allreduce(dolfinx.fem.assemble_scalar(stim_form), op=MPI.SUM), 0.0)


def test_generate_random_activation():
    """Tests the spatial and temporal activation logic of the UFL expression."""
    # 1. Setup a basic 3D mesh and time constant
    domain = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    t = dolfinx.fem.Constant(domain, 0.0)

    # 2. Define parameters
    # Point 1 activates at t = 1.0, ends at t = 2.0
    # Point 2 activates at t = 3.0, ends at t = 4.0
    points = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    delays = np.array([1.0, 3.0])
    stim_start = 0.0
    stim_duration = 1.0
    stim_amplitude = 5.0

    # Use a larger tolerance for the test so we guarantee it hits interpolation points
    tol = 0.2

    # 3. Generate the expression
    stim_expr = beat.stimulation.generate_random_activation(
        mesh=domain,
        time=t,
        points=points,
        delays=delays,
        stim_start=stim_start,
        stim_duration=stim_duration,
        stim_amplitude=stim_amplitude,
        tol=tol,
    )

    # 4. Set up a Function Space (DG0) to evaluate the expression on the mesh cells
    V = dolfinx.fem.functionspace(domain, ("DG", 0))
    expr = dolfinx.fem.Expression(stim_expr, beat.utils.interpolation_points(V))
    stim_func = dolfinx.fem.Function(V)

    # --- Verify Time Stepping ---

    # Case A: Before any activation (t = 0.5)
    t.value = 0.5
    stim_func.interpolate(expr)
    assert np.allclose(stim_func.x.array, 0.0), "Expected 0.0 everywhere before first delay."

    # Case B: First point is active (t = 1.5)
    t.value = 1.5
    stim_func.interpolate(expr)
    assert np.max(stim_func.x.array) == pytest.approx(
        stim_amplitude,
    ), "Expected first point to activate."
    assert np.min(stim_func.x.array) == pytest.approx(
        0.0,
    ), "Expected the rest of the mesh to remain 0.0."

    # Case C: Gap between activations (t = 2.5)
    t.value = 2.5
    stim_func.interpolate(expr)
    assert np.allclose(stim_func.x.array, 0.0), "Expected 0.0 everywhere between activations."

    # Case D: Second point is active (t = 3.5)
    t.value = 3.5
    stim_func.interpolate(expr)
    assert np.max(stim_func.x.array) == pytest.approx(
        stim_amplitude,
    ), "Expected second point to activate."

    # Case E: After all activations are finished (t = 4.5)
    t.value = 4.5
    stim_func.interpolate(expr)
    assert np.allclose(stim_func.x.array, 0.0), "Expected 0.0 everywhere after all durations end."


def test_generate_random_activation_assertion():
    """Tests that mismatched array lengths raise the expected AssertionError."""
    domain = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    t = dolfinx.fem.Constant(domain, 0.0)

    points = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    delays = np.array([1.0])  # Intentionally mismatched length

    with pytest.raises(AssertionError, match="Points and delays must have the same length"):
        beat.stimulation.generate_random_activation(domain, t, points, delays)


def test_generate_random_activation_recursion():
    """
    Tests that a large number of stimulation points doesn't trigger a
    RecursionError when UFL builds and traverses the expression tree.
    """
    # Ensure the standard recursion limit is set
    import sys

    sys.setrecursionlimit(1000)

    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 2, 2, 2)
    time = dolfinx.fem.Constant(mesh, 0.0)

    # 1500 points would previously guarantee a RecursionError
    # (Default depth is 1000)
    num_points = 1500
    points = np.random.rand(num_points, 3)
    delays = np.random.rand(num_points)

    expr = beat.stimulation.generate_random_activation(
        mesh=mesh,
        time=time,
        points=points,
        delays=delays,
        stim_start=0.0,
        stim_duration=2.0,
        stim_amplitude=1.0,
        tol=1e-12,
    )

    # Trigger UFL to traverse the tree (e.g., by converting it to a string representation)
    # If the tree is too deep, this will raise a RecursionError.
    try:
        _ = str(expr)
    except RecursionError:
        pytest.fail("generate_random_activation raised RecursionError on AST traversal")
