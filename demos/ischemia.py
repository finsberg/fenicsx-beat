# # Acute myocardial ischemia at the cellular level
# This demo shows the effects of mild hyperkalemia and reduced
# sodium conductance and calcium permeability on the cardiac action potential.
#
# Three cellular conditions are considered:
#
# - **Healthy:** the default ToR-ORd endocardial-cell parameters.
# - **Border zone:** mild extracellular hyperkalemia together with reduced
#   sodium conductance and calcium permeability.
# - **Ischemic core:** more severe extracellular hyperkalemia and stronger
#   reductions in sodium conductance and calcium permeability.
#

from pathlib import Path
import time
import numba
from scipy.integrate import solve_ivp
import re
import matplotlib.pyplot as plt
import numpy as np
import gotranx

def set_param(model, params, name, value):
    try:
        idx = model["parameter_index"](name)
    except Exception:
        print(f"Warning: parameter {name} not found")
        return params
    params[idx] = value
    return params

#setting the ischemia parameters for the model

def make_ischemia_parameters(model, cellType, severity):
    p = model["init_parameter_values"](celltype=cellType)

    if severity == "healthy":
        return p

    if severity == "border":
        set_param(model, p, "ko", 7.0)                # Mild Hyperkalemia
        set_param(model, p, "GNa", 10.01317)          # 85% Sodium Conductance
        set_param(model, p, "PCa_b", 6.70056e-05)     # 80% Calcium Permeability
        return p

    if severity == "core":
        set_param(model, p, "ko", 10.0)               # Severe Hyperkalemia             
        set_param(model, p, "GNa", 8.83515)           # 75% Sodium Conductance
        set_param(model, p, "PCa_b", 6.281775e-05)    # 75% Calcium Permeability
        return p

    raise ValueError(f"Unknown severity: {severity}")

def get_monitor_indices(model, monitor_names):
    indices = {}
    if "monitor_values" not in model or "monitor_index" not in model:
        print("No monitor_values / monitor_index found in generated model.", flush=True)
        print("Model keys containing 'monitor':", flush=True)
        print([k for k in model.keys() if "monitor" in k], flush=True)
        return indices

    for name in monitor_names:
        try:
            indices[name] = model["monitor_index"](name)
        except Exception:
            print(f"Warning: monitor {name} not found", flush=True)

    return indices
def call_monitor_values(model, t, y, p):
    monitor_fun = model["monitor_values"]

    for args in (
        (t, y, p),
        (y, t, p),
    ):
        try:
            return monitor_fun(*args)
        except TypeError:
            pass

    for kwargs in (
        {"states": y, "t": t, "parameters": p},
        {"states": y, "time": t, "parameters": p},
    ):
        try:
            return monitor_fun(**kwargs)
        except TypeError:
            pass

    return monitor_fun(t, y, p)
def evaluate_monitors(model, t_values, y_values, p, monitor_indices, stride=1):
    if not monitor_indices:
        return {}, np.array([])

    idx_values = np.arange(0, len(t_values), stride)
    t_monitor = t_values[idx_values]

    monitor_data = {name: np.zeros(len(idx_values)) for name in monitor_indices}

    for j, i in enumerate(idx_values):
        monitors = call_monitor_values(model, t_values[i], y_values[:, i], p)
        for name, idx in monitor_indices.items():
            monitor_data[name][j] = monitors[idx]

    return monitor_data, t_monitor
def get_param_value(model, p, name, default=np.nan):
    try:
        return float(p[model["parameter_index"](name)])
    except Exception:
        print(f"Warning: parameter {name} not found")
        return default
def parse_named_values(ode, block_name, index_function):
    """
    Parse state/parameter names from the ODE file and sort them by the
    generated model's index function. This makes the saved state matrix
    self-describing.
    """
    text = ode.read_text()
    pattern = re.compile(
        rf'{block_name}\("[^"]+"\s*,(.*?)\n\)',
        flags=re.DOTALL,
    )

    names = []
    for block in pattern.findall(text):
        names.extend(
            re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", block, re.MULTILINE)
        )

    indexed = []
    for name in names:
        try:
            indexed.append((index_function(name), name))
        except Exception:
            pass

    indexed.sort(key=lambda item: item[0])
    return [name for _, name in indexed]

def main():

    demo_directory = Path(__file__).resolve().parent
    repository_root = demo_directory.parent

    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    print("Running model")
    # Load the model
    model_path = demo_directory / "ToRORd_dynCl_endo.py"
    ode_file = (
        repository_root
        / "odes"
        / "torord"
        / "ToRORd_dynCl_endo.ode"
    )
    if not model_path.is_file():
        print("Generate code for cell model")
        
        ode = gotranx.load_ode(ode_file)
        code = gotranx.cli.gotran2py.get_code(
            ode,
            scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
        )
        model_path.write_text(code)

    import ToRORd_dynCl_endo

    model = ToRORd_dynCl_endo.__dict__
    



    state_names_all = parse_named_values(
        ode_file,
        "states",
        model["state_index"],
    )

    parameter_names_all = parse_named_values(
        ode_file,
        "parameters",
        model["parameter_index"],
    )

    print(f"Number of states: {len(state_names_all)}")
    print(f"Number of parameters: {len(parameter_names_all)}")


    # ------------------------------------------------------------
    # Choose which cell types to run
    # ------------------------------------------------------------
    celltypes_to_run = {
        "endo": 0,
       
    }

    # ------------------------------------------------------------
    # Get variable indices
    # ------------------------------------------------------------
    V_index = model["state_index"]("v")
    Ca_index = model["state_index"]("cai")
    

    rhs = numba.njit(model["rhs"])

    # ------------------------------------------------------------
    # All membrane currents appearing in dv_dt
    # ------------------------------------------------------------
    current_names = [
        "INa_INa",
        "INaL_INaL",
        "Ito_Ito",

        "ICaL_ICaL",
        "ICaNa",
        "ICaK",

        "IKr_IKr",
        "IKs_IKs",
        "IK1_IK1",

        "INaCa_i",
        "INaCa_ss",
        "INaK_INaK",

        "INab_INab",
        "IKb_IKb",
        "IpCa_IpCa",
        "ICab_ICab",

        "IClCa",
        "IClb",

        "I_katp_I_katp",
        "Istim",
    ]

    # Additional non-current monitor needed for ATP calculations
    extra_monitor_names = [
        "Jup",
    ]

    monitor_names = current_names + extra_monitor_names

    monitor_indices = get_monitor_indices(model, monitor_names)
    # ------------------------------------------------------------
    # Simulation settings
    # ------------------------------------------------------------
    save_all_beats = False
    reuse_checkpoint = True

    conditioning_beats = 10
    BCL = 1000.0
    dt = 0.01
    save_frequency = 1

    # Store 0 <= t < BCL. Include exactly BCL only in the solver output
    # so the final column can be used as the restart state.
    times = np.arange(0.0, BCL, dt * save_frequency)
    solver_times = np.append(times, BCL)

    t0_all = time.perf_counter()

    for celltype_name, cell_type in celltypes_to_run.items():
        # This dictionary must persist while healthy, border and core run.
        voltage_traces = {}

        for severity in ("healthy", "border", "core"):
            print("=" * 70, flush=True)
            print(
                f"Running celltype: {celltype_name}, severity: {severity}",
                flush=True,
            )
            print("=" * 70, flush=True)

            t0_severity = time.perf_counter()

            checkpoint_file = (
                outdir
                / f"Torord_checkpoint_{celltype_name}_{severity}"
                f"_nbeats{conditioning_beats}"
                f"_BCL{BCL}_dt{dt}.npz"
            )

            all_beats_file = (
                outdir
                / f"Torord_all_beats_all_states_"
                f"{celltype_name}_{severity}"
                f"_nbeats{conditioning_beats}"
                f"_BCL{BCL}_dt{dt}.npy"
            )

            y = None
            p = None

            # --------------------------------------------------------
            # Load a compatible checkpoint when available
            # --------------------------------------------------------
            if reuse_checkpoint and checkpoint_file.is_file():
                with np.load(
                    checkpoint_file,
                    allow_pickle=False,
                ) as checkpoint:
                    required_keys = {
                        "final_state",
                        "parameters",
                        "conditioning_beats",
                        "BCL",
                        "dt",
                        "celltype",
                        "severity",
                    }

                    checkpoint_is_compatible = required_keys.issubset(
                        checkpoint.files
                    )

                    if checkpoint_is_compatible:
                        checkpoint_is_compatible = (
                            int(checkpoint["conditioning_beats"])
                            == conditioning_beats
                            and np.isclose(
                                float(checkpoint["BCL"]),
                                BCL,
                            )
                            and np.isclose(
                                float(checkpoint["dt"]),
                                dt,
                            )
                            and str(checkpoint["celltype"].item())
                            == celltype_name
                            and str(checkpoint["severity"].item())
                            == severity
                        )

                    if checkpoint_is_compatible:
                        y = checkpoint["final_state"].copy()
                        p = checkpoint["parameters"].copy()

                        print(
                            f"Loaded compatible checkpoint: "
                            f"{checkpoint_file}",
                            flush=True,
                        )
                    else:
                        print(
                            "Checkpoint exists but is incompatible; "
                            f"regenerating: {checkpoint_file}",
                            flush=True,
                        )

            # A final-state checkpoint cannot reconstruct the full history.
            # If that history is requested and absent, conditioning must run.
            need_all_beats_history = (
                save_all_beats and not all_beats_file.is_file()
            )

            if need_all_beats_history and y is not None:
                print(
                    "The checkpoint is reusable, but the all-beats archive "
                    "is missing. Conditioning will run again to record it.",
                    flush=True,
                )

            # --------------------------------------------------------
            # Condition only when no valid checkpoint is available
            # --------------------------------------------------------
            if y is None or need_all_beats_history:
                y = model["init_state_values"]()
                p = make_ischemia_parameters(
                    model,
                    cell_type,
                    severity,
                )

                all_beats_states = None
                all_beats_time = None
                temporary_states_file = None
                temporary_time_file = None

                if save_all_beats:
                    samples_per_beat = len(times)
                    total_samples = (
                        conditioning_beats * samples_per_beat
                    )
                    number_of_states = len(state_names_all)

                    temporary_states_file = (
                        outdir
                        / f".temporary_all_states_"
                        f"{celltype_name}_{severity}.npy"
                    )
                    temporary_time_file = (
                        outdir
                        / f".temporary_all_times_"
                        f"{celltype_name}_{severity}.npy"
                    )

                    all_beats_states = np.lib.format.open_memmap(
                        temporary_states_file,
                        mode="w+",
                        dtype=np.float64,
                        shape=(number_of_states, total_samples),
                    )
                    all_beats_time = np.lib.format.open_memmap(
                        temporary_time_file,
                        mode="w+",
                        dtype=np.float64,
                        shape=(total_samples,),
                    )

                for beat in range(conditioning_beats):
                    t1 = time.perf_counter()

                    result = solve_ivp(
                        rhs,
                        (0.0, BCL),
                        y,
                        t_eval=(
                            solver_times
                            if save_all_beats
                            else [BCL]
                        ),
                        method="BDF",
                        args=(p,),
                    )

                    if not result.success:
                        raise RuntimeError(
                            f"Conditioning failed for "
                            f"{celltype_name} {severity}, "
                            f"beat {beat + 1}: "
                            f"{result.message}"
                        )

                    if save_all_beats:
                        start = beat * samples_per_beat
                        stop = start + samples_per_beat

                        all_beats_time[start:stop] = (
                            beat * BCL + times
                        )
                        all_beats_states[:, start:stop] = (
                            result.y[:, :-1]
                        )

                    y = result.y[:, -1].copy()

                    print(
                        f"{celltype_name} | {severity} | "
                        f"conditioning beat "
                        f"{beat + 1}/{conditioning_beats} | "
                        f"elapsed = "
                        f"{time.perf_counter() - t1:.2f} s",
                        flush=True,
                    )

                np.savez_compressed(
                    checkpoint_file,
                    final_state=y.copy(),
                    parameters=p.copy(),
                    state_names=np.asarray(
                        state_names_all,
                        dtype=str,
                    ),
                    parameter_names=np.asarray(
                        parameter_names_all,
                        dtype=str,
                    ),
                    conditioning_beats=np.asarray(
                        conditioning_beats
                    ),
                    BCL=np.asarray(BCL),
                    dt=np.asarray(dt),
                    celltype=np.asarray(celltype_name),
                    severity=np.asarray(severity),
                )

                print(
                    f"Checkpoint saved: {checkpoint_file}",
                    flush=True,
                )

                if save_all_beats:
                    all_beats_states.flush()
                    all_beats_time.flush()

                    all_beats_result = {
                        "time_ms": np.asarray(all_beats_time),
                        "states": np.asarray(all_beats_states),
                        "state_names": np.asarray(
                            state_names_all,
                            dtype=str,
                        ),
                        "final_state": y.copy(),
                        "parameters": p.copy(),
                        "parameter_names": np.asarray(
                            parameter_names_all,
                            dtype=str,
                        ),
                        "BCL": BCL,
                        "dt": dt,
                        "nbeats": conditioning_beats,
                        "celltype": celltype_name,
                        "severity": severity,
                    }

                    np.save(
                        all_beats_file,
                        all_beats_result,
                        allow_pickle=True,
                    )

                    del all_beats_states
                    del all_beats_time

                    temporary_states_file.unlink(
                        missing_ok=True
                    )
                    temporary_time_file.unlink(
                        missing_ok=True
                    )

                    print(
                        f"Complete all-beats archive saved: "
                        f"{all_beats_file}",
                        flush=True,
                    )

            # --------------------------------------------------------
            # Record one additional detailed beat for THIS severity
            # --------------------------------------------------------
            t1 = time.perf_counter()

            result = solve_ivp(
                rhs,
                (0.0, BCL),
                y,
                t_eval=solver_times,
                method="BDF",
                args=(p,),
            )

            if not result.success:
                raise RuntimeError(
                    f"Detailed beat failed for "
                    f"{celltype_name} {severity}: "
                    f"{result.message}"
                )

            last_beat_time = result.t[:-1].copy()
            last_beat_all_states = result.y[:, :-1].copy()
            restart_state = result.y[:, -1].copy()

            Vs = last_beat_all_states[V_index, :].copy()
            Cais = last_beat_all_states[Ca_index, :].copy()

            # This assignment is inside the severity loop and therefore
            # executes once for healthy, border and core.
            voltage_traces[severity] = {
                "time_ms": last_beat_time.copy(),
                "V": Vs.copy(),
            }

            print(
                f"Stored voltage trace for {severity}. "
                f"Available traces: {list(voltage_traces)}",
                flush=True,
            )

            monitor_data, monitor_time = evaluate_monitors(
                model,
                last_beat_time,
                last_beat_all_states,
                p,
                monitor_indices,
                stride=1,
            )

            current_traces = {
                name: monitor_data[name]
                for name in current_names
                if name in monitor_data
            }

            n_total = len(last_beat_time)
            A_atp = np.full(
                n_total,
                get_param_value(
                    model,
                    p,
                    "A_atp",
                    np.nan,
                ),
            )
            MgATP = np.full(
                n_total,
                get_param_value(
                    model,
                    p,
                    "MgATP",
                    np.nan,
                ),
            )
            MgADP = np.full(
                n_total,
                get_param_value(
                    model,
                    p,
                    "MgADP",
                    np.nan,
                ),
            )

            last_beat_file = (
                outdir
                / f"Torord_last_beat_all_states_"
                f"{celltype_name}_{severity}"
                f"_BCL{BCL}_dt{dt}.npz"
            )

            np.savez_compressed(
                last_beat_file,
                time_ms=last_beat_time,
                states=last_beat_all_states,
                state_names=np.asarray(
                    state_names_all,
                    dtype=str,
                ),
                final_state=restart_state,
                parameters=p.copy(),
                parameter_names=np.asarray(
                    parameter_names_all,
                    dtype=str,
                ),
                BCL=np.asarray(BCL),
                dt=np.asarray(dt),
                nbeats=np.asarray(1),
                celltype=np.asarray(celltype_name),
                severity=np.asarray(severity),
            )

            trace_result = {
                "time_ms": last_beat_time,
                "V": Vs,
                "cai": Cais,
                "A_atp": A_atp,
                "MgATP": MgATP,
                "MgADP": MgADP,
                "currents": current_traces,
                "monitors": monitor_data,
                "monitor_time_ms": monitor_time,
                "beat_summary": [],
                "final_state": restart_state,
                "parameters": p.copy(),
                "celltype": celltype_name,
                "severity": severity,
                "BCL": BCL,
                "dt": dt,
                "nbeats": 1,
                "current_names": list(
                    current_traces.keys()
                ),
                "monitor_names_found": list(
                    monitor_indices.keys()
                ),
            }

            trace_file = (
                outdir
                / f"Torord_trace_{celltype_name}_{severity}"
                f"_nbeats1_BCL{BCL}.npy"
            )

            np.save(
                trace_file,
                trace_result,
                allow_pickle=True,
            )

            print(
                f"Saved complete last beat: "
                f"{last_beat_file}",
                flush=True,
            )
            print(
                f"Trace saved: {trace_file}",
                flush=True,
            )
            print(
                f"{celltype_name} | {severity} | "
                f"detailed beat elapsed = "
                f"{time.perf_counter() - t1:.2f} s",
                flush=True,
            )
            print(
                f"{celltype_name} {severity} total elapsed: "
                f"{time.perf_counter() - t0_severity:.2f} s",
                flush=True,
            )

        # ------------------------------------------------------------
        # Plot only after all three severity loops have finished
        # ------------------------------------------------------------
        expected_severities = {"healthy", "border", "core"}
        available_severities = set(voltage_traces)

        print(
            f"Voltage traces ready for plotting: "
            f"{list(voltage_traces)}",
            flush=True,
        )

        if available_severities != expected_severities:
            missing = expected_severities - available_severities
            raise RuntimeError(
                "Cannot create the voltage comparison plot. "
                f"Missing traces: {sorted(missing)}"
            )

        fig, ax = plt.subplots(figsize=(8, 5))

        for severity in ("healthy", "border", "core"):
            trace = voltage_traces[severity]
            ax.plot(
                trace["time_ms"],
                trace["V"],
                linewidth=2.0,
                label=severity.capitalize(),
            )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Membrane potential (mV)")
        ax.set_title(
            "Effect of ischemia on the action potential"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        voltage_figure = (
            outdir
            / f"Torord_voltage_comparison_{celltype_name}"
            f"_BCL{BCL}_dt{dt}.png"
        )

        fig.savefig(
            voltage_figure,
            dpi=300,
            bbox_inches="tight",
        )

        # plt.show()
        plt.close(fig)

        print(
            f"Voltage comparison saved: {voltage_figure}",
            flush=True,
        )

    print(
        f"Total simulation elapsed: "
        f"{time.perf_counter() - t0_all:.2f} s",
        flush=True,
    )
if __name__ == "__main__":
    main()
