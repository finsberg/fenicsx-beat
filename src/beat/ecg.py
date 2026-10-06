import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

from petsc4py import PETSc

import dolfinx
import numpy as np
import ufl
from numpy.typing import ArrayLike
from packaging.version import Version

logger = logging.getLogger(__name__)


from scipy.signal import find_peaks

_dolfinx_version = Version(dolfinx.__version__)


def detect_r_peaks(ecg_signal: np.ndarray, min_distance: float = 20) -> np.ndarray:
    """
    Detects R-peaks in the ECG signal.

    Parameters
    ----------
    ecg_signal : numpy.ndarray
        The ECG signal to be processed. Should be filtered.
    min_distance : float, optional
        Minimum distance between R-peaks in milliseconds. Defaults to 20 ms.

    Returns
    -------
    numpy.ndarray
        Indices of the detected R-peaks in the ECG signal.

    """

    # Add a height threshold relative to the signal's max value to avoid noisy peaks
    height_threshold = 0.5 * np.max(ecg_signal) if np.max(ecg_signal) > 0 else None

    peaks, _ = find_peaks(ecg_signal, distance=min_distance, height=height_threshold)
    return peaks


def detect_t_end(
    averaged_rr: np.ndarray,
    r_peak_index: int,
    window_start_offset: int = 50,
    window_end_offset: int = 400,
) -> int:
    """
    Detects the end of the T-wave in the averaged RR interval using a derivative-based method.

    Parameters
    ----------
    averaged_rr : numpy.ndarray
        The averaged RR interval. Should not be None.
    r_peak_index : int
        The index of the R-peak in the averaged RR interval.
    window_start_offset : int, optional
        Start of the search window relative to R-peak (milliseconds). Defaults to 50.
    window_end_offset : int, optional
        End of the search window relative to R-peak (milliseconds). Defaults to 400.

    Returns
    -------
    int
        Index of the T-wave end relative to the start of averaged_rr.
    """

    if averaged_rr is None or len(averaged_rr) == 0:
        raise RuntimeError("Error: Cannot detect T-end on empty or None averaged RR interval.")

    # Define the search window for the T-wave end based on offsets from R-peak
    search_start = r_peak_index + window_start_offset  # * sampling_rate)
    search_end = r_peak_index + window_end_offset  # * sampling_rate)

    # Ensure indices are within the bounds of the averaged_rr array
    search_start = max(0, search_start)  # Ensure start is not negative
    search_end = min(len(averaged_rr), search_end)  # Ensure end does not exceed array length

    # Check if the search window is valid
    if (
        search_start >= search_end or search_end - search_start < 2
    ):  # Need at least 2 points for diff
        logger.warning("Invalid or too short search window for T-end detection.")
        # return None

    # Extract the segment for T-wave end detection
    signal_segment = averaged_rr[search_start:search_end]

    # Calculate the first derivative (velocity) of the signal segment
    derivative = np.diff(signal_segment)

    if len(derivative) == 0:
        logger.warning("Could not compute derivative for T-end detection (segment too short?).")

        # return None

    # Find T-peak index within the segment (relative to segment start)
    t_peak_index_relative = int(
        np.argmax(np.abs(signal_segment)),
    )  # Find peak of T-wave (can be positive or negative)

    # Search for T-end after the T-peak
    search_start_tend = t_peak_index_relative  # Start search slightly after T-peak
    search_start_tend = max(0, search_start_tend)  # Ensure start is not negative

    if search_start_tend >= len(derivative):
        logger.warning("T-peak is too close to the end of the search window.")
        # return None

    # Find the point where the derivative returns close to zero after the T-peak
    # This is a simplified approach; tangent methods are more common in literature
    # Find the index of the minimum derivative *after* the T-peak
    min_derivative_after_peak_index = np.argmin(derivative[search_start_tend:])

    # Calculate T-end relative index
    t_end_index_relative = search_start_tend + min_derivative_after_peak_index

    # Convert relative index back to the original averaged_rr index
    t_end_index_absolute = search_start + t_end_index_relative

    # Basic validation: T-end should be after R-peak

    if t_end_index_absolute <= r_peak_index:
        logger.warning("Detected T-end is before or at the R-peak index.")
        # return None

    return int(t_end_index_absolute)


# def correct_qt_interval(
#     qt_interval_ms: float,
#     rr_interval_duration_s: float,
#     method: Literal["bazett", "fridericia"] = "bazett",
# ):
#     """
#     Corrects the QT interval for heart rate using Bazett's or Fridericia's formula.

#     Parameters:
#     ----------
#     qt_interval_ms : float
#         The QT interval in milliseconds. Can be None.
#     rr_interval_duration_s : float
#         The RR interval duration in seconds.
#     method : str, optional
#         The correction method ('bazett' or 'fridericia'). Defaults to 'bazett'.

#     Returns:
#     -------
#     float
#         The corrected QT interval (QTc) in milliseconds. Returns None if input is invalid.

#     """

#     qt_interval_s = qt_interval_ms / 1000.0  # Convert QT to seconds for formula

#     if method.lower() == "bazett":
#         # QTc = QT / sqrt(RR)
#         qtc_s = qt_interval_s / np.sqrt(rr_interval_duration_s)
#     elif method.lower() == "fridericia":
#         # QTc = QT / cubic-sqrt(RR)
#         qtc_s = qt_interval_s / (rr_interval_duration_s ** (1 / 3))
#     else:
#         raise ValueError(
#             f"Invalid QTc correction method '{method}'. Use 'bazett' or 'fridericia'.",
#         )

#     qtc_ms = qtc_s * 1000.0  # Convert back to milliseconds
#     return qtc_ms


class QTIntervalResult(NamedTuple):
    qt_interval: float
    start_index: int
    end_index: int


def qt_interval(
    t: np.ndarray,
    ecg_signal: np.ndarray,
    min_distance: float = 20.0,
    window_start_offset: int = 50,
    window_end_offset: int = 400,
) -> QTIntervalResult:
    """
    Processes the ECG signal to compute the corrected QT interval (QTc).

    Parameters:
    ----------
    t : np.ndarray
        Time vector corresponding to the ECG signal in seconds.
    ecg_signal : np.ndarray
        The ECG signal to be processed. Should be filtered.
    min_distance : float, optional
        Minimum distance between R-peaks in seconds. Defaults to 20 ms.
    window_start_offset : int, optional
        Start of the search window for T-wave end relative to R-peak (milliseconds). Defaults to 50.
    window_end_offset : int, optional
        End of the search window for T-wave end relative to R-peak (milliseconds). Defaults to 400.

    Returns:
    -------
    QTIntervalResult
        A named tuple containing the start index, end index, and the QT interval in seconds.
        Returns None if no R-peaks are detected or if the T-end cannot be determined.
    """

    r_peaks = detect_r_peaks(ecg_signal=ecg_signal, min_distance=min_distance)
    assert len(r_peaks) > 0, "No R-peaks detected. Check signal quality and detection parameters."
    r_peak_index = r_peaks[0]
    t_end_index = detect_t_end(
        ecg_signal,
        r_peak_index,
        window_start_offset=window_start_offset,
        window_end_offset=window_end_offset,
    )

    qt_interval = t[t_end_index] - t[r_peak_index]

    return QTIntervalResult(
        start_index=r_peak_index,
        end_index=t_end_index,
        qt_interval=qt_interval,
    )


@dataclass
class ECGRecovery:
    v: dolfinx.fem.Function
    sigma_b: float | dolfinx.fem.Constant = 1.0
    C_m: float | dolfinx.fem.Constant = 1.0
    dx: ufl.Measure | None = None
    M: float = 1.0
    petsc_options: dict[str, Any] = field(
        default_factory=lambda: {
            "ksp_type": "cg",
            "pc_type": "sor",
            # "ksp_monitor": None,
            "ksp_rtol": 1.0e-8,
            "ksp_atol": 1.0e-8,
            # "ksp_error_if_not_converged": True,
        },
    )

    def __post_init__(self):
        if self.dx is None:
            self.dx = ufl.dx(domain=self.mesh, metadata={"quadrature_degree": 4})
        self.sol = dolfinx.fem.Function(self.V)

        w = ufl.TestFunction(self.V)
        Im = ufl.TrialFunction(self.V)

        self.sol = dolfinx.fem.Function(self.V)

        self._lhs = -self.C_m * Im * w * self.dx
        self._rhs = ufl.inner(self.M * ufl.grad(self.v), ufl.grad(w)) * self.dx

        kwargs = {}
        if _dolfinx_version >= Version("0.10"):
            kwargs["petsc_options_prefix"] = "beat_ecg_recovery_"

        self.solver = dolfinx.fem.petsc.LinearProblem(
            self._lhs,
            self._rhs,
            u=self.sol,
            petsc_options=self.petsc_options,
            **kwargs,
        )
        dolfinx.fem.petsc.assemble_matrix(self.solver.A, self.solver.a)
        self.solver.A.assemble()

    @property
    def V(self) -> dolfinx.fem.FunctionSpace:
        return self.v.function_space

    @property
    def mesh(self) -> dolfinx.mesh.Mesh:
        return self.v.function_space.mesh

    def solve(self):
        logger.debug("Solving ECG recovery")
        with self.solver.b.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(self.solver.b, self.solver.L)
        self.solver.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )

        self.solver.solver.solve(self.solver.b, self.sol.x.petsc_vec)
        self.sol.x.scatter_forward()

    def eval(self, point) -> dolfinx.fem.forms.Form:
        r = ufl.SpatialCoordinate(self.mesh) - dolfinx.fem.Constant(self.mesh, point)
        dist = ufl.sqrt((r**2))
        return dolfinx.fem.form((1 / (4 * ufl.pi * self.sigma_b)) * (self.sol / dist) * self.dx)


@dataclass(frozen=True)
class Lead:
    """One lead: a named linear combination of potentials, ``Σ c·φ[p]``."""

    name: str
    terms: Mapping[str, float]


@dataclass(frozen=True)
class LeadSystem:
    """A set of leads, as data.

    Parameters
    ----------
    definitions : tuple[Lead, ...]
        The leads, in output order.
    derived : Mapping[str, Mapping[str, float]]
        Points that are not electrodes, each a weighted sum of electrode
        positions, ``Σ w·x``. The terms of a lead may refer to them.
    layout : tuple[tuple[str, ...], ...] | None
        Optional plotting layout, one tuple of lead names per row.
    """

    definitions: tuple[Lead, ...]
    derived: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    layout: tuple[tuple[str, ...], ...] | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(lead.name for lead in self.definitions)

    @property
    def electrodes(self) -> tuple[str, ...]:
        """The electrodes needed: the non-derived points in the terms, plus those the
        derived points' weights use, in order of first appearance."""
        seen: dict[str, None] = {}
        for lead in self.definitions:
            for p in lead.terms:
                if p in self.derived:
                    for e in self.derived[p]:
                        seen.setdefault(e)
                else:
                    seen.setdefault(p)
        return tuple(seen)

    def points(self, electrodes: Mapping[str, ArrayLike]) -> dict[str, np.ndarray]:
        """Every given electrode as a float array, then each derived point as ``Σ w·x``."""
        missing = [e for e in self.electrodes if e not in electrodes]
        if missing:
            raise ValueError(f"Missing electrode positions: {', '.join(missing)}")
        out = {k: np.asarray(v, dtype=float) for k, v in electrodes.items()}
        for name, weights in self.derived.items():
            terms = [w * out[e] for e, w in weights.items()]
            out[name] = sum(terms[1:], terms[0])
        return out

    def leads(
        self,
        potentials: Mapping[str, float | np.ndarray],
    ) -> dict[str, float | np.ndarray]:
        """Each lead's ``Σ c·φ[p]``, in ``definitions`` order."""
        needed: dict[str, None] = {}
        for lead in self.definitions:
            for p in lead.terms:
                needed.setdefault(p)
        missing = [p for p in needed if p not in potentials]
        if missing:
            raise ValueError(f"Missing potentials: {', '.join(missing)}")
        out: dict[str, float | np.ndarray] = {}
        for lead in self.definitions:
            terms = [c * potentials[p] for p, c in lead.terms.items()]
            out[lead.name] = sum(terms[1:], terms[0])
        return out


# (lead, positive electrode, negative pole), after
# https://en.wikipedia.org/wiki/Electrocardiography
_TWELVE_LEAD_TABLE: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("I", "LA", ("RA",)),
    ("II", "LL", ("RA",)),
    ("III", "LL", ("LA",)),
    ("aVR", "RA", ("LA", "LL")),
    ("aVL", "LA", ("RA", "LL")),
    ("aVF", "LL", ("RA", "LA")),
    *((f"V{i}", f"V{i}", ("LA", "RA", "LL")) for i in range(1, 7)),
)
_TWELVE_LEAD_DERIVED = {
    ("LA", "RA", "LL"): "WCT_pt",
    ("LA", "LL"): "LA_LL_mid",
    ("RA", "LL"): "RA_LL_mid",
    ("RA", "LA"): "RA_LA_mid",
}
_TWELVE_LEAD_LAYOUT = (
    ("I", "aVR", "V1", "V4"),
    ("II", "aVL", "V2", "V5"),
    ("III", "aVF", "V3", "V6"),
)


def twelve_lead(reference: Literal["potential", "position"] = "potential") -> LeadSystem:
    """The standard 12-lead ECG (Einthoven, Goldberger, Wilson).

    Parameters
    ----------
    reference : {"potential", "position"}
        ``"potential"``: a negative pole of n electrodes contributes ``-1/n`` of each
        electrode's potential (the textbook leads). ``"position"``: the legacy simcardems
        variant, where a negative pole of more than one electrode is a single point at
        the mean of their positions (``WCT_pt``, ``LA_LL_mid``, ``RA_LL_mid``,
        ``RA_LA_mid``), and the lead is a difference of two potentials.
    """
    if reference not in ("potential", "position"):
        raise ValueError(f"reference must be 'potential' or 'position', got {reference!r}")
    leads = []
    used: dict[str, Mapping[str, float]] = {}
    for name, positive, pole in _TWELVE_LEAD_TABLE:
        n = len(pole)
        if reference == "potential" or n == 1:
            terms = {positive: 1.0, **{e: -1.0 / n for e in pole}}
        else:
            pt = _TWELVE_LEAD_DERIVED[tuple(pole)]
            used[pt] = {e: 1.0 / n for e in pole}
            terms = {positive: 1.0, pt: -1.0}
        leads.append(Lead(name, terms))
    return LeadSystem(tuple(leads), derived=used, layout=_TWELVE_LEAD_LAYOUT)


def example(
    sampling_rate_hz: int = 1000,
    duration_s: float = 10,
    heart_rate_bpm: float = 60,
    q_offset_ms: float = 40,
    s_offset_ms: float = 40,
    t_peak_offset_ms: float = 200,
    r_width_ms: float = 20,
    q_width_ms: float = 20,
    s_width_ms: float = 30,
    t_width_ms: float = 60,
    qrs_peak_time: float = 200,
    noise_amplitude: float = 0.0,
    wander_freq_hz: float = 0.2,
    wander_amplitude: float = 0.1,
):
    """
    Generate a synthetic ECG signal.

    Parameters
    ----------
    sampling_rate_hz : int
        Sampling rate in Hz.
    duration_s : float
        Duration of the signal in seconds.
    heart_rate_bpm : float
        Heart rate in beats per minute.
    q_offset_ms : float
        Offset for the Q wave in milliseconds.
    s_offset_ms : float
        Offset for the S wave in milliseconds.
    t_peak_offset_ms : float
        Offset for the T peak in milliseconds.
    r_width_ms : float
        Width of the R wave in milliseconds.
    q_width_ms : float
        Width of the Q wave in milliseconds.
    s_width_ms : float
        Width of the S wave in milliseconds.
    t_width_ms : float
        Width of the T wave in milliseconds.
    qrs_peak_time : float
        Start time for the qrs peak time in milliseconds.
    noise_amplitude : float
        Amplitude of the noise to be added to the signal.
    wander_freq_hz : float
        Frequency of the baseline wander in Hz.
    wander_amplitude : float
        Amplitude of the baseline wander.

    Returns
    -------
    t_ms : np.ndarray
        Time vector in milliseconds.
    ecg_signal : np.ndarray
        Generated ECG signal.
    """

    # Convert time parameters to milliseconds
    duration_ms = duration_s * 1000
    rr_interval_s = 60.0 / heart_rate_bpm
    rr_interval_ms = rr_interval_s * 1000

    num_beats = int(duration_s / rr_interval_s)

    # Time vector in milliseconds
    num_samples = int(duration_s * sampling_rate_hz)
    t_ms = np.linspace(0, duration_ms, num_samples, endpoint=False)

    ecg_signal = np.zeros_like(t_ms)

    # Create multiple beats
    for i in range(num_beats):
        # R-peak time for the current beat, in milliseconds
        r_peak_time_ms = (i + qrs_peak_time / 1000) * rr_interval_ms

        # Calculate absolute times for other wave components for this beat
        q_time_ms = r_peak_time_ms - q_offset_ms
        s_time_ms = r_peak_time_ms + s_offset_ms
        t_peak_time_ms = r_peak_time_ms + t_peak_offset_ms

        # Add waves for the current beat
        # R peak
        ecg_signal += 1.0 * np.exp(-(((t_ms - r_peak_time_ms) / r_width_ms) ** 2))
        # Q wave
        ecg_signal -= 0.2 * np.exp(-(((t_ms - q_time_ms) / q_width_ms) ** 2))
        # S wave
        ecg_signal -= 0.3 * np.exp(-(((t_ms - s_time_ms) / s_width_ms) ** 2))
        # T wave
        ecg_signal += 0.4 * np.exp(-(((t_ms - t_peak_time_ms) / t_width_ms) ** 2))

    # Add some baseline noise
    if noise_amplitude > 0:
        ecg_signal += noise_amplitude * np.random.randn(len(t_ms))

    # Add some baseline wander (low frequency noise)
    wander_freq_per_ms = wander_freq_hz / 1000.0

    ecg_signal += wander_amplitude * np.sin(2 * np.pi * wander_freq_per_ms * t_ms)

    return t_ms, ecg_signal
