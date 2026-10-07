import threading
from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

import beat.c_backend

comm = MPI.COMM_WORLD

# A two-state forward Euler step with a time-dependent term, in the gotranx per-cell signature
TOY_C = r"""
#include <math.h>
void toy(const double *__restrict states, const double t, const double dt,
         const double *__restrict parameters, double *values)
{
    values[0] = states[0] - parameters[0] * states[1] * dt;
    values[1] = states[1] + parameters[1] * states[0] * dt + t * dt;
}
"""


def toy_numpy(states, t, parameters, dt):
    values = np.empty_like(states)
    values[0] = states[0] - parameters[0] * states[1] * dt
    values[1] = states[1] + parameters[1] * states[0] * dt + t * dt
    return values


@pytest.fixture
def cache_dir(tmp_path):
    # All ranks must use rank 0's directory
    return Path(comm.bcast(str(tmp_path), root=0))


@pytest.fixture
def toy(cache_dir):
    return beat.c_backend.compile_scheme(
        TOY_C,
        "toy",
        num_states=2,
        num_parameters=2,
        cache_dir=cache_dir,
    )


def random_states(n_points, seed=1):
    return np.random.default_rng(seed).uniform(-1, 1, size=(2, n_points))


def test_matches_numpy_2d(toy):
    states = random_states(37)
    parameters = np.array([1.5, 0.5])
    result = toy(states=states, t=0.3, parameters=parameters, dt=0.1)
    assert result.shape == states.shape
    assert result is not states
    np.testing.assert_allclose(result, toy_numpy(states, 0.3, parameters, 0.1), rtol=1e-14)


def test_matches_numpy_1d_states(toy):
    states = random_states(1)[:, 0].copy()
    parameters = np.array([1.5, 0.5])
    result = toy(states=states, t=0.3, parameters=parameters, dt=0.1)
    np.testing.assert_allclose(result, toy_numpy(states, 0.3, parameters, 0.1), rtol=1e-14)


def test_spatially_varying_parameters(toy):
    states = random_states(11)
    parameters = np.random.default_rng(2).uniform(0, 2, size=(2, 11))
    result = toy(states=states, t=0.0, parameters=parameters, dt=0.1)
    np.testing.assert_allclose(result, toy_numpy(states, 0.0, parameters, 0.1), rtol=1e-14)


def test_out_argument(toy):
    states = random_states(5)
    parameters = np.array([1.0, 1.0])
    out = np.zeros_like(states)
    result = toy(states=states, t=0.0, parameters=parameters, dt=0.1, out=out)
    assert result is out
    np.testing.assert_allclose(out, toy_numpy(states, 0.0, parameters, 0.1), rtol=1e-14)


def test_out_aliasing_states(toy):
    states = random_states(9)
    expected = toy_numpy(states, 0.2, np.array([1.0, 2.0]), 0.1)
    toy(states=states, t=0.2, parameters=np.array([1.0, 2.0]), dt=0.1, out=states)
    np.testing.assert_allclose(states, expected, rtol=1e-14)


@pytest.mark.parametrize(
    "states, parameters, error",
    [
        (np.zeros((2, 4), dtype=np.float32), np.ones(2), TypeError),
        (np.zeros((4, 2)).T, np.ones(2), ValueError),  # not C-contiguous
        (np.zeros((2, 8))[:, ::2], np.ones(2), ValueError),  # strided
        (np.zeros((2, 4)), np.array([1, 1]), TypeError),  # int parameters
        (np.zeros((2, 4, 1)), np.ones(2), ValueError),  # 3D
        (np.zeros((2, 4)), np.ones((2, 3)), ValueError),  # parameter points mismatch
        ([[0.0], [0.0]], np.ones(2), TypeError),  # not an ndarray
    ],
)
def test_rejects_bad_arrays(toy, states, parameters, error):
    with pytest.raises(error):
        toy(states=states, t=0.0, parameters=parameters, dt=0.1)


def test_rejects_bad_out(toy):
    with pytest.raises(ValueError):
        toy(states=np.zeros((2, 4)), t=0.0, parameters=np.ones(2), dt=0.1, out=np.zeros((2, 5)))


def test_wrong_number_of_states_or_parameters(toy):
    with pytest.raises(ValueError, match="states"):
        toy(states=np.zeros((3, 4)), t=0.0, parameters=np.ones(2), dt=0.1)
    with pytest.raises(ValueError, match="parameters"):
        toy(states=np.zeros((2, 4)), t=0.0, parameters=np.ones(5), dt=0.1)


def test_library_in_cache_dir(toy, cache_dir):
    assert toy.library_path.is_file()
    assert toy.library_path.parent.parent == cache_dir.resolve()
    assert toy.scheme == "toy"


def test_rank0_os_error_raises_on_all_ranks(cache_dir):
    # A regular file where a directory is needed makes mkdir fail on rank 0 (chmod does not
    # work as root). Every rank must raise instead of the others hanging in the broadcast.
    blocker = cache_dir / "blocker"
    if comm.rank == 0:
        blocker.write_text("")
    comm.barrier()
    with pytest.raises(RuntimeError, match="Error"):
        beat.c_backend.compile_scheme(
            TOY_C,
            "toy",
            num_states=2,
            num_parameters=2,
            cache_dir=blocker / "sub",
        )


def count_compiles(monkeypatch):
    calls = []
    original = beat.c_backend._run_compiler

    def counting(cmd):
        if "-shared" in cmd:
            calls.append(cmd)
        return original(cmd)

    monkeypatch.setattr(beat.c_backend, "_run_compiler", counting)
    return calls


def test_cache_hit_does_not_recompile(cache_dir, monkeypatch):
    calls = count_compiles(monkeypatch)
    first = beat.c_backend.compile_scheme(TOY_C, "toy", cache_dir=cache_dir)
    second = beat.c_backend.compile_scheme(TOY_C, "toy", cache_dir=cache_dir)
    assert first.library_path == second.library_path
    if comm.rank == 0:
        assert len(calls) == 1
    else:
        assert len(calls) == 0


def test_flags_change_key(cache_dir):
    a = beat.c_backend.compile_scheme(TOY_C, "toy", cache_dir=cache_dir, cflags=("-O2",))
    b = beat.c_backend.compile_scheme(TOY_C, "toy", cache_dir=cache_dir, cflags=("-O3",))
    assert a.library_path != b.library_path


def test_env_cache_dir(cache_dir, monkeypatch):
    monkeypatch.setenv("BEAT_C_CACHE", str(cache_dir / "env"))
    fun = beat.c_backend.compile_scheme(TOY_C, "toy")
    assert (cache_dir / "env").resolve() in fun.library_path.parents


def test_missing_compiler_raises_on_all_ranks(cache_dir):
    with pytest.raises(RuntimeError, match="does-not-exist-cc"):
        beat.c_backend.compile_scheme(TOY_C, "toy", cache_dir=cache_dir, cc="does-not-exist-cc")


def test_compile_error_raises_with_stderr(cache_dir):
    with pytest.raises(RuntimeError, match="Compiling the C scheme failed"):
        beat.c_backend.compile_scheme("void toy(this is not C", "toy", cache_dir=cache_dir)
    # A failed compile leaves no library behind
    if comm.rank == 0:
        assert not list(cache_dir.rglob(beat.c_backend.LIBRARY_NAME))


def test_unknown_scheme_name_raises(cache_dir):
    with pytest.raises(RuntimeError, match="Compiling the C scheme failed"):
        beat.c_backend.compile_scheme(TOY_C, "not_a_function", cache_dir=cache_dir)


def test_concurrent_compiles_of_same_key(tmp_path):
    source = beat.c_backend.full_source(TOY_C, "toy")
    directory = tmp_path / "key"
    results, errors = [], []

    def work():
        try:
            results.append(beat.c_backend._compile(source, "cc", ("-O2",), directory))
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(set(results)) == 1
    fun = beat.c_backend.CScheme(results[0], "toy")
    states = random_states(3)
    np.testing.assert_allclose(
        fun(states=states, t=0.0, parameters=np.ones(2), dt=0.1),
        toy_numpy(states, 0.0, np.ones(2), 0.1),
        rtol=1e-14,
    )
    # Only the library and the source are left, no temporary files
    assert sorted(p.name for p in directory.iterdir()) == [beat.c_backend.LIBRARY_NAME, "scheme.c"]


def test_check_visible_missing_library_raises_on_all_ranks(tmp_path):
    with pytest.raises(RuntimeError, match="BEAT_C_CACHE"):
        beat.c_backend._check_visible(tmp_path / "missing.so", comm)


ODE_FILE = (
    Path(__file__).parents[1]
    / "odes"
    / "tentusscher_panfilov_2006"
    / "tentusscher_panfilov_2006_epi_cell.ode"
)


@pytest.fixture(scope="module")
def tp06(tmp_path_factory):
    gotranx = pytest.importorskip("gotranx")
    ode = gotranx.load_ode(ODE_FILE)
    code = gotranx.cli.gotran2py.get_code(
        ode,
        scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    )
    namespace: dict = {}
    exec(compile(code, "tentusscher_panfilov_2006_epi_cell", "exec"), namespace)
    cache = Path(comm.bcast(str(tmp_path_factory.mktemp("c_cache")), root=0))
    fun = beat.c_backend.from_ode(ode, scheme="generalized_rush_larsen", cache_dir=cache)
    return ode, namespace, fun


def test_from_ode_matches_numpy_grl(tp06):
    ode, model, fun = tp06
    assert fun.num_states == ode.num_states
    assert fun.num_parameters == ode.num_parameters
    parameters = model["init_parameter_values"]()  # stimulus on: covers the upstroke
    states_np = np.zeros((ode.num_states, 20))
    states_np.T[:] = model["init_state_values"]()
    states_c = states_np.copy()
    dt, t = 0.05, 0.0
    for _ in range(100):
        states_np = model["generalized_rush_larsen"](states_np, t, dt, parameters)
        states_c = fun(states=states_c, t=t, parameters=parameters, dt=dt)
        t += dt
    np.testing.assert_allclose(states_c, states_np, rtol=1e-10, atol=1e-12)


def test_c_helpers_match_numpy_model(tp06):
    ode, model, fun = tp06
    assert fun.state_index("V") == model["state_index"]("V")
    assert fun.parameter_index("g_Kr") == model["parameter_index"]("g_Kr")
    np.testing.assert_allclose(fun.init_state_values(), model["init_state_values"]())
    np.testing.assert_allclose(fun.init_parameter_values(), model["init_parameter_values"]())
    with pytest.raises(KeyError):
        fun.state_index("not_a_state")
    with pytest.raises(KeyError):
        fun.parameter_index("not_a_parameter")


def test_init_values_need_counts(toy):
    unknown = beat.c_backend.CScheme(toy.library_path, "toy")
    with pytest.raises(ValueError, match="num_states"):
        unknown.init_state_values()


def test_from_ode_unknown_scheme_raises_on_all_ranks(tp06, cache_dir):
    ode, _, _ = tp06
    with pytest.raises(RuntimeError):
        beat.c_backend.from_ode(ode, scheme="not_a_scheme", cache_dir=cache_dir)


def test_target_signature_only_for_native_flags():
    # Non-native flags should return empty string
    result = beat.c_backend.target_signature("cc", ("-O3",))
    assert result == ""

    # Native flags should return a non-empty string containing architecture macros
    result = beat.c_backend.target_signature("cc", ("-O3", "-march=native"))
    assert result != ""
    assert "__x86_64__" in result or "__aarch64__" in result


def test_target_signature_changes_key(cache_dir, monkeypatch):
    # Monkeypatch target_signature on all ranks
    def fake_target_signature_a(cc, flags):
        return (
            beat.c_backend.target_signature(cc, flags)
            if not any("native" in f for f in flags)
            else "cpu-A"
        )

    def fake_target_signature_b(cc, flags):
        return (
            beat.c_backend.target_signature(cc, flags)
            if not any("native" in f for f in flags)
            else "cpu-B"
        )

    # Compile with fake signature A
    monkeypatch.setattr(
        beat.c_backend,
        "target_signature",
        fake_target_signature_a,
    )
    lib_a = beat.c_backend.compile_scheme(
        TOY_C,
        "toy",
        num_states=2,
        num_parameters=2,
        cache_dir=cache_dir,
        cflags=("-O2", "-march=native"),
    )

    # Compile with fake signature B
    monkeypatch.setattr(
        beat.c_backend,
        "target_signature",
        fake_target_signature_b,
    )
    lib_b = beat.c_backend.compile_scheme(
        TOY_C,
        "toy",
        num_states=2,
        num_parameters=2,
        cache_dir=cache_dir,
        cflags=("-O2", "-march=native"),
    )

    # The two libraries should be in different cache directories due to different keys
    assert lib_a.library_path != lib_b.library_path


def run_monodomain(fun, model, num_states):
    import dolfinx
    import ufl

    import beat

    mesh = dolfinx.mesh.create_unit_square(comm, 10, 10, dolfinx.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    tdim = mesh.topology.dim
    cells = dolfinx.mesh.locate_entities(
        mesh,
        tdim,
        lambda x: np.logical_and(x[0] <= 0.2 + 1e-10, x[1] <= 0.2 + 1e-10),
    )
    tags = dolfinx.mesh.meshtags(mesh, tdim, cells, np.full(len(cells), 1, dtype=np.int32))
    dx = ufl.dx(domain=mesh, subdomain_data=tags)
    stim = ufl.conditional(ufl.le(time, 2.0), dolfinx.fem.Constant(mesh, 100.0), 0.0)
    pde = beat.MonodomainModel(
        time=time,
        mesh=mesh,
        M=0.1,
        I_s=beat.Stimulus(expr=stim, dZ=dx, marker=1),
    )
    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1))),
        v_pde=pde.state,
        fun=fun,
        init_states=model["init_state_values"](),
        parameters=model["init_parameter_values"](stim_amplitude=0.0),
        num_states=num_states,
        v_index=model["state_index"]("V"),
    )
    solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)
    t, dt = 0.0, 0.05
    for _ in range(60):
        solver.step((t, t + dt))
        t += dt
    return pde.state.x.array.copy()


def test_monodomain_c_matches_numpy(tp06):
    ode, model, fun = tp06
    v_numpy = run_monodomain(model["generalized_rush_larsen"], model, ode.num_states)
    v_c = run_monodomain(fun, model, ode.num_states)
    # Reduce before asserting so that every rank reaches both collectives
    v_peak = comm.allreduce(v_numpy.max(initial=-np.inf), op=MPI.MAX)
    difference = comm.allreduce(np.max(np.abs(v_c - v_numpy), initial=0.0), op=MPI.MAX)
    assert v_peak > 0.0  # the stimulus produced an action potential
    assert difference <= 1e-9
