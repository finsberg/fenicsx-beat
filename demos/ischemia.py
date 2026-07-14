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
# ## Simulation workflow
#
# The simulation is divided into two stages:
#
# 1. ``checkpoint`` mode conditions each cellular model for a specified
#    number of pacing cycles and stores the final ODE state.
# 2. ``last_beat`` mode loads the corresponding checkpoint, simulates one
#    additional beat, stores the complete state trajectory and selected
#    membrane-current traces, and compares the action potentials.
#3. ``save_all_beats`` mode simulates a specified number of beats
#    starting from the initial state, recording and saving the complete
#    state trajectory and membrane currents for *every* single beat,
#    allowing to study transient conditioning dynamics.
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
    # Simulation settings & Choose run mode
    # ------------------------------------------------------------
    
    # Save the complete time history of every ODE state for all
    # conditioning beats. The final state is also stored separately
    # in the compact checkpoint file.

    save_all_beats = False  # or True 

    # First run: create conditioned checkpoint
    run_mode = "checkpoint"

    # Second run: load checkpoint and save one detailed beat

    # run_mode = "last_beat"

    conditioning_beats = 10
    
    BCL = 1000.0
    dt = 0.01
    save_frequency = 1

    if run_mode == "checkpoint":
        nbeats = conditioning_beats
    elif run_mode == "last_beat":
        nbeats = 1
    else:
        raise ValueError(
            f"Unknown run_mode: {run_mode}"
        )

    # Time points stored in output trajectories. The exact BCL endpoint is
    # excluded to prevent duplication between consecutive beats.

    times = np.arange(0.0, BCL, dt * save_frequency)

    # Include the exact beat endpoint when solving so that the restart state
    # corresponds to t = BCL.

    solver_times = np.append(times, BCL)

    all_times = np.arange(0.0, BCL * nbeats, dt * save_frequency)
    t0_all = time.perf_counter()

    for celltype_name, cell_type in celltypes_to_run.items():
        voltage_traces = {}

        for severity in ["healthy", "border", "core"]:
        # for severity in ["healthy"]:
            print("=" * 70, flush=True)
            print(f"Running celltype: {celltype_name}, severity: {severity}", flush=True)
            print("=" * 70, flush=True)

            checkpoint_file = (
                outdir
                / f"Torord_checkpoint_{celltype_name}_{severity}"
                f"_BCL{BCL}_dt{dt}.npz"
            )

            if run_mode == "last_beat":
                if not checkpoint_file.is_file():
                    raise FileNotFoundError(
                        f"Checkpoint not found: {checkpoint_file}\n"
                        "Run once with run_mode = 'checkpoint'."
                    )

                checkpoint = np.load(
                    checkpoint_file,
                    allow_pickle=False,
                )

                y = checkpoint["final_state"].copy()
                p = checkpoint["parameters"].copy()

                print(
                    f"Loaded checkpoint: {checkpoint_file}",
                    flush=True,
                )

            else:
                y = model["init_state_values"]()

                p = make_ischemia_parameters(
                    model,
                    cell_type,
                    severity,
                )

            last_beat_all_states = None
            last_beat_time = None
            restart_state = None

            Vs = None
            Cais = None
            A_atp = None
            MgATP = None
            MgADP = None
            monitor_traces = None
            beat_summary = []
  
            if run_mode == "last_beat":
                n_total = len(times) * nbeats
     
                Vs = np.zeros(len(times) * nbeats)
                Cais = np.zeros(len(times) * nbeats)
            
                A_atp_value = get_param_value(model, p, "A_atp", np.nan)
                MgATP_value = get_param_value(model, p, "MgATP", np.nan)
                MgADP_value = get_param_value(model, p, "MgADP", np.nan)

                A_atp = np.full(n_total, A_atp_value)
                MgATP = np.full(n_total, MgATP_value)
                MgADP = np.full(n_total, MgADP_value)

                monitor_traces = {
                    name: np.zeros(n_total)
                    for name in monitor_indices
                }
                       
            t0 = time.perf_counter()

            all_beats_states = None
            all_beats_time = None
            all_beats_file = None

            if run_mode == "checkpoint" and save_all_beats:
                samples_per_beat = len(times)
                total_samples = nbeats * samples_per_beat
                number_of_states = len(state_names_all)

                all_beats_file = (
                    outdir
                    / f"Torord_all_beats_all_states_"
                    f"{celltype_name}_{severity}"
                    f"_nbeats{nbeats}_BCL{BCL}_dt{dt}.npy"
                )

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

                print(
                    f"Recording all states for {nbeats} beats",
                    flush=True,
                )
            for beat in range(nbeats):

                # ============================================================
                # FAST CHECKPOINT MODE
                # ============================================================
                if run_mode == "checkpoint":
                    t1 = time.perf_counter()

                    # Include BCL so that the restart state corresponds exactly to
                    # the end of the beat. The endpoint is not duplicated in the
                    # saved all-beats trajectory.
                   

                    res = solve_ivp(
                        rhs,
                        (0.0, BCL),
                        y,
                        t_eval=solver_times if save_all_beats else [BCL],
                        method="BDF",
                        args=(p,),
                    )

                    if not res.success:
                        raise RuntimeError(
                            f"solve_ivp failed for "
                            f"{celltype_name} {severity}, "
                            f"beat {beat + 1}: {res.message}"
                        )

                    
                    if save_all_beats:
                        # The final column corresponds to exactly BCL and is used as
                        # the restart state. It is excluded from the stored trajectory
                        # because the next beat starts at the same physical time.

                        beat_states = res.y[:, :-1]

                        start = beat * samples_per_beat
                        stop = start + samples_per_beat

                        all_beats_time[start:stop] = (
                            beat * BCL + times
                        )

                        all_beats_states[:, start:stop] = beat_states
                    # Restart from the state at exactly t = BCL.
                    
                    y = res.y[:, -1].copy()

            

                    print(
                        f"{celltype_name} | {severity} | "
                        f"checkpoint beat {beat + 1}/{nbeats} | "
                        f"elapsed = "
                        f"{time.perf_counter() - t1:.2f} s",
                        flush=True,
                    )
                    continue
          
                # ============================================================
                # ONE DETAILED BEAT
                # ============================================================
                t1 = time.perf_counter()

                res = solve_ivp(
                    rhs,
                    (0.0, BCL),
                    y,
                    t_eval=solver_times,
                    method="BDF",
                    args=(p,),
                )

                if not res.success:
                    raise RuntimeError(
                        f"solve_ivp failed for "
                        f"{celltype_name} {severity}, "
                        f"beat {beat + 1}: {res.message}"
                    )

                # Saved trajectory excludes the duplicate endpoint

                last_beat_time = res.t[:-1].copy()
                last_beat_all_states = res.y[:, :-1].copy()


                # Save the state from which another beat can start

                restart_state = res.y[:, -1].copy()

                Vs[:] = last_beat_all_states[V_index, :]
                Cais[:] = last_beat_all_states[Ca_index, :]

                voltage_traces[severity] = {
                    "time_ms": last_beat_time.copy(),
                    "V": last_beat_all_states[V_index, :].copy(),
                }


                monitor_data, monitor_time = evaluate_monitors(
                    model,
                    last_beat_time,
                    last_beat_all_states,
                    p,
                    monitor_indices,
                    stride=1,
                )

                for name, arr in monitor_data.items():
                    monitor_traces[name][:] = arr

                y = restart_state.copy()

                print(
                    f"{celltype_name:5s} | "
                    f"{severity:8s} | "
                    f"beat {beat + 1}/{nbeats} | "
                    f"elapsed = "
                    f"{time.perf_counter() - t1:.2f} s",
                    flush=True,
                )           

            # ================================================================
            # SAVE ALL CONDITIONING BEATS
            # ================================================================
            if (
                run_mode == "checkpoint"
                and save_all_beats
                and all_beats_states is not None
                and all_beats_time is not None
            ):
                all_beats_states.flush()
                all_beats_time.flush()

                print(
                    f"Writing complete all-beats archive: "
                    f"{all_beats_file}",
                    flush=True,
                )

                all_beats_result = {
                    # Complete time vector for all conditioning beats
                    "time_ms": np.asarray(all_beats_time),

                    # Every ODE state at every saved time point
                    "states": np.asarray(all_beats_states),

                    # State names corresponding to rows of `states`
                    "state_names": np.asarray(
                        state_names_all,
                        dtype=str,
                    ),

                    # State at the exact end of the final conditioning beat
                    "final_state": y.copy(),

                    # Complete parameter vector and parameter names
                    "parameters": p.copy(),

                    "parameter_names": np.asarray(
                        parameter_names_all,
                        dtype=str,
                    ),

                    # Simulation metadata
                    "BCL": BCL,
                    "dt": dt,
                    "nbeats": nbeats,
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

                all_beats_states = None
                all_beats_time = None

                temporary_states_file.unlink(missing_ok=True)
                temporary_time_file.unlink(missing_ok=True)

                print(
                    f"Complete all-beats archive saved: "
                    f"{all_beats_file}",
                    flush=True,
                )

            print(
                f"{celltype_name} {severity} total elapsed: "
                f"{time.perf_counter() - t0:.2f} s",
                flush=True,
            )      

            # ================================================================
            # SAVE CONDITIONING CHECKPOINT
            # ================================================================
            if run_mode == "checkpoint":

                np.savez_compressed(
                    checkpoint_file,

                    # Complete final ODE state
                    final_state=y.copy(),

                    # Complete parameter vector
                    parameters=p.copy(),

                    # State and parameter names
                    state_names=np.array(
                        state_names_all,
                        dtype=str,
                    ),

                    parameter_names=np.array(
                        parameter_names_all,
                        dtype=str,
                    ),

                    BCL=np.array(BCL),
                    dt=np.array(dt),
                    conditioning_beats=np.array(
                        conditioning_beats
                    ),
                    celltype=np.array(celltype_name),
                    severity=np.array(severity),
                )

                print(
                    f"Checkpoint saved: {checkpoint_file}",
                    flush=True,
                )

            # ================================================================
            # SAVE ONE COMPLETE DETAILED BEAT
            # ================================================================
            elif run_mode == "last_beat":

                if (
                    last_beat_all_states is None
                    or last_beat_time is None
                    or restart_state is None
                ):
                    raise RuntimeError(
                        "Detailed beat was not recorded."
                    )

                last_beat_file = (
                    outdir
                    / f"Torord_last_beat_all_states_"
                    f"{celltype_name}_{severity}"
                    f"_BCL{BCL}_dt{dt}.npz"
                )

                np.savez_compressed(
                    last_beat_file,

                    # Complete time vector
                    time_ms=last_beat_time,

                    # Every ODE state at every time point
                    states=last_beat_all_states,

                    # Names corresponding to state matrix rows
                    state_names=np.array(
                        state_names_all,
                        dtype=str,
                    ),

                    # State to restart another beat
                    final_state=restart_state,

                    # Complete parameter vector and names
                    parameters=p.copy(),

                    parameter_names=np.array(
                        parameter_names_all,
                        dtype=str,
                    ),

                    BCL=np.array(BCL),
                    dt=np.array(dt),
                    nbeats=np.array(1),
                    celltype=np.array(celltype_name),
                    severity=np.array(severity),
                )

                print(
                    f"Saved complete last beat: "
                    f"{last_beat_file}",
                    flush=True,
                )

                # Keep membrane currents in their own dictionary
                current_traces = {
                    name: monitor_traces[name]
                    for name in current_names
                    if name in monitor_traces
                }

                trace_result = {
                    "time_ms": all_times,
                    "V": Vs,
                    "cai": Cais,

                    # Constant ATP-related ToR-ORd parameters
                    "A_atp": A_atp,
                    "MgATP": MgATP,
                    "MgADP": MgADP,

                    # All membrane currents from dv_dt
                    "currents": current_traces,

                    # Includes currents plus Jup
                    "monitors": monitor_traces,

                    # Estimated ATP demands
                    "beat_summary": beat_summary,

                    # Complete final state and parameters
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

                    # "atp_term_names": atp_term_names,
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
                    f"Trace saved: {trace_file}",
                    flush=True,
                )

        #Plotting voltage traces for all severities
        if run_mode == "last_beat":
            expected_severities = {"healthy", "border", "core"}
            available_severities = set(voltage_traces)

            if available_severities != expected_severities:
                missing = expected_severities - available_severities
                raise RuntimeError(
                    "Cannot create the voltage comparison plot. "
                    f"Missing traces: {sorted(missing)}"
                )

            fig, ax = plt.subplots(figsize=(8, 5))

            for severity in ("healthy", "border", "core"):
            # for severity in ["healthy"]:
                trace = voltage_traces[severity]

                ax.plot(
                    trace["time_ms"],
                    trace["V"],
                    linewidth=2.0,
                    label=severity.capitalize(),
                )

            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Membrane potential (mV)")
            ax.set_title("Effect of ischemia on the action potential")
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

if __name__ == "__main__":
    main()
