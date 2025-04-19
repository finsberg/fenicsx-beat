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
