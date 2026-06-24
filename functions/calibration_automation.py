"""Automation helpers for Bragg calibration notebooks.

The functions here are intentionally notebook-friendly: they accept the arrays
already used in the calibration notebooks and return pandas tables that can be
plotted, inspected, saved, and compared against hand-selected reference points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks, peak_widths


@dataclass(frozen=True)
class EnergyCalibration:
    """Linear spectrometer energy-axis correction used by the notebooks."""

    ecentral: float = 12407.0
    dEdP: float = 0.105
    dispersion_original: float = 0.15


@dataclass(frozen=True)
class PeakDetectionConfig:
    """Settings for image-only peak candidate extraction.

    brightness_frac:
        Minimum peak intensity as a fraction of the *global* image maximum.
        Mirrors the manual pipeline's ``thresh = spec_hist.max() / 17`` gate.
        Set to 0 to disable (not recommended — disabling is what made the
        original codex implementation emit 14 000+ noise peaks per scan).
    use_global_sigma:
        If True, estimate one noise scale from the whole image rather than a
        per-column rolling MAD.  Prevents the local MAD from collapsing near
        the bright seed line and promoting surrounding noise.
    """

    background_window: int = 51
    noise_window: int = 101
    prominence_sigma: float = 10.0
    use_global_sigma: bool = True
    brightness_frac: float = 1.0 / 17.0
    min_width_px: float | None = None
    max_width_px: float | None = None
    max_peaks_per_spectrum: int = 3
    min_separation_px: int = 3


@dataclass(frozen=True)
class RidgeLinkConfig:
    """Greedy model-free ridge linking settings."""

    max_sample_gap: int = 3
    max_energy_gap: float = 20.0
    max_slope_change: float | None = None
    min_points: int = 6


@dataclass(frozen=True)
class RidgeFilterConfig:
    """Post-linking filters for candidate ridges.

    The defaults intentionally reject short, nearly horizontal spectral bands
    while keeping sloped Bragg-like traces.

    top_n_ridges:
        If set, keep only the top-N ridges ranked by
        ``integrated_prominence × angle_span × |slope_dE_dangle|``.  This
        enforces the physics prior that there are O(10) real Bragg lines per
        scan; set to None to skip.
    """

    min_points: int = 6
    min_angle_span: float = 0.0
    min_sample_span: int = 5
    min_energy_span: float = 3.0
    min_abs_slope_dE_dangle: float | None = 15.0
    max_gap_fraction: float = 0.5
    top_n_ridges: int | None = 20


def corrected_energy_axis(
    phen_scale: Sequence[float],
    calibration: EnergyCalibration = EnergyCalibration(),
) -> np.ndarray:
    """Return the corrected energy axis used in the current notebook."""

    phen = np.asarray(phen_scale, dtype=float)
    return (phen - calibration.ecentral) * calibration.dEdP / calibration.dispersion_original + calibration.ecentral


def scan_type_from_doocs_label(label: Any) -> str:
    """Classify a DOOCS channel label as pitch, roll, or unknown."""

    text = str(label)
    if "MONOPA" in text:
        return "pitch"
    if "MONORA" in text:
        return "roll"
    return "unknown"


def load_scan_table(
    filepath: str | Path,
    *,
    calibration: EnergyCalibration = EnergyCalibration(),
    use_corrected_energy: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load one `.npz` scan into a sample table plus spectra/image arrays.

    Returns
    -------
    sample_table:
        One row per spectrum/sample.
    spectra:
        Array shaped `(n_samples, n_energy)`, usually `spec_hist`.
    image:
        Array shaped approximately `(n_angle_bins, n_energy)`, from `corr2d`.
    """

    path = Path(filepath)
    data = np.load(path, allow_pickle=True)
    spectra = np.asarray(data["spec_hist"], dtype=float)
    doocs_vals = np.asarray(data["doocs_vals_hist"], dtype=float)
    phen_scale = np.asarray(data["phen_scale"], dtype=float)
    energy = corrected_energy_axis(phen_scale, calibration) if use_corrected_energy else phen_scale
    label = data["doocs_channel"]
    scan_type = scan_type_from_doocs_label(label)

    if spectra.ndim != 2:
        raise ValueError(f"Expected spec_hist to be 2D, got shape {spectra.shape}")
    if spectra.shape[0] != doocs_vals.size:
        raise ValueError(
            "spec_hist sample count does not match doocs_vals_hist: "
            f"{spectra.shape[0]} != {doocs_vals.size}"
        )
    if spectra.shape[1] != energy.size:
        raise ValueError(
            "spec_hist energy axis does not match phen_scale: "
            f"{spectra.shape[1]} != {energy.size}"
        )

    sample_table = pd.DataFrame(
        {
            "scan_id": path.stem,
            "filepath": str(path),
            "sample_index": np.arange(spectra.shape[0], dtype=int),
            "scan_type": scan_type,
            "angle": doocs_vals,
            "energy_min": float(np.nanmin(energy)),
            "energy_max": float(np.nanmax(energy)),
            "doocs_channel": str(label),
        }
    )
    return sample_table, spectra, np.asarray(data["corr2d"], dtype=float)


def _robust_sigma(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    med = np.nanmedian(values, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(values - med), axis=axis)
    sigma = 1.4826 * mad
    return np.maximum(sigma, np.finfo(float).eps)


def _odd_window(window: int, max_size: int) -> int:
    window = int(max(3, window))
    window = min(window, max_size if max_size % 2 else max_size - 1)
    if window % 2 == 0:
        window += 1
    return max(3, window)


def detect_peak_candidates(
    spectra: np.ndarray,
    energy_axis: Sequence[float],
    angles: Sequence[float],
    *,
    scan_id: str = "",
    config: PeakDetectionConfig = PeakDetectionConfig(),
) -> pd.DataFrame:
    """Detect image-only peak candidates in every spectrum.

    This stage deliberately does not use Bragg curves or H-plane identities.
    """

    spectra = np.asarray(spectra, dtype=float)
    energy = np.asarray(energy_axis, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if spectra.ndim != 2:
        raise ValueError(f"Expected spectra to be 2D, got shape {spectra.shape}")
    if spectra.shape[1] != energy.size:
        raise ValueError("Energy axis length must match spectra.shape[1]")
    if spectra.shape[0] != angles.size:
        raise ValueError("Angles length must match spectra.shape[0]")

    bg_window = _odd_window(config.background_window, spectra.shape[1])
    noise_window = _odd_window(config.noise_window, spectra.shape[1])
    background = ndimage.median_filter(spectra, size=(1, bg_window), mode="nearest")
    residual = spectra - background

    if config.use_global_sigma:
        # One noise scale for the whole image prevents the local MAD from
        # collapsing next to the bright seed line and promoting surrounding noise.
        global_sigma = float(_robust_sigma(residual.ravel()))
        sigma_map = np.full(spectra.shape, max(global_sigma, np.finfo(float).eps))
    else:
        sigma_by_spectrum = _robust_sigma(residual, axis=1)
        local_mad = ndimage.median_filter(np.abs(residual), size=(1, noise_window), mode="nearest")
        sigma_map = np.maximum(1.4826 * local_mad, sigma_by_spectrum[:, None])

    normalized = residual / sigma_map
    abs_threshold = config.brightness_frac * float(np.nanmax(spectra)) if config.brightness_frac > 0 else -np.inf

    rows: list[dict[str, Any]] = []
    for sample_index, norm_spectrum in enumerate(normalized):
        peaks, props = find_peaks(
            norm_spectrum,
            prominence=config.prominence_sigma,
            distance=config.min_separation_px,
        )
        if peaks.size == 0:
            continue

        widths, _, left_ips, right_ips = peak_widths(norm_spectrum, peaks, rel_height=0.5)
        order = np.argsort(props["prominences"])[::-1]
        kept = 0
        for pos in order:
            peak_idx = int(peaks[pos])
            # Absolute-brightness gate: the seeded Bragg line is one of the
            # globally brightest features in the image; noise peaks are not.
            if spectra[sample_index, peak_idx] < abs_threshold:
                continue
            width_px = float(widths[pos])
            if config.min_width_px is not None and width_px < config.min_width_px:
                continue
            if config.max_width_px is not None and width_px > config.max_width_px:
                continue
            rows.append(
                {
                    "scan_id": scan_id,
                    "sample_index": int(sample_index),
                    "angle": float(angles[sample_index]),
                    "energy": float(energy[peak_idx]),
                    "energy_index": peak_idx,
                    "intensity": float(spectra[sample_index, peak_idx]),
                    "background": float(background[sample_index, peak_idx]),
                    "snr": float(norm_spectrum[peak_idx]),
                    "prominence": float(props["prominences"][pos]),
                    "width_px": width_px,
                    "left_energy_index": float(left_ips[pos]),
                    "right_energy_index": float(right_ips[pos]),
                    "sigma_E": _estimate_sigma_e(energy, width_px),
                }
            )
            kept += 1
            if kept >= config.max_peaks_per_spectrum:
                break

    columns = [
        "scan_id",
        "sample_index",
        "angle",
        "energy",
        "energy_index",
        "intensity",
        "background",
        "snr",
        "prominence",
        "width_px",
        "left_energy_index",
        "right_energy_index",
        "sigma_E",
    ]
    return pd.DataFrame(rows, columns=columns)


def _estimate_sigma_e(energy_axis: np.ndarray, width_px: float) -> float:
    if energy_axis.size < 2:
        return np.nan
    step = float(np.nanmedian(np.abs(np.diff(energy_axis))))
    fwhm = max(float(width_px) * step, step)
    return fwhm / 2.355


def link_ridge_candidates(
    peaks: pd.DataFrame,
    *,
    config: RidgeLinkConfig = RidgeLinkConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily link peak candidates into model-free ridge candidates."""

    if peaks.empty:
        ridge_points = peaks.copy()
        ridge_points["ridge_id"] = pd.Series(dtype="int64")
        ridges = _empty_ridge_table()
        return ridge_points, ridges

    required = {"scan_id", "sample_index", "angle", "energy", "prominence"}
    missing = required.difference(peaks.columns)
    if missing:
        raise ValueError(f"Missing required peak columns: {sorted(missing)}")

    points = peaks.reset_index(drop=False).rename(columns={"index": "peak_id"})
    points = points.sort_values(["scan_id", "sample_index", "energy"]).reset_index(drop=True)
    points["ridge_id"] = -1

    active: dict[str, list[dict[str, Any]]] = {}
    next_ridge = 0

    for row_idx, row in points.iterrows():
        scan_id = row["scan_id"]
        sample_index = int(row["sample_index"])
        candidates = active.setdefault(scan_id, [])
        best_idx = None
        best_cost = np.inf
        for active_idx, ridge in enumerate(candidates):
            gap = sample_index - int(ridge["last_sample"])
            if gap <= 0 or gap > config.max_sample_gap:
                continue
            predicted_energy = float(ridge["last_energy"])
            if ridge["slope"] is not None:
                predicted_energy += float(ridge["slope"]) * gap
            energy_gap = abs(float(row["energy"]) - predicted_energy)
            if energy_gap > config.max_energy_gap:
                continue
            if config.max_slope_change is not None and ridge["slope"] is not None:
                new_slope = (float(row["energy"]) - float(ridge["last_energy"])) / gap
                if abs(new_slope - float(ridge["slope"])) > config.max_slope_change:
                    continue
            cost = energy_gap / max(float(row.get("prominence", 1.0)), 1.0)
            if cost < best_cost:
                best_cost = cost
                best_idx = active_idx

        if best_idx is None:
            ridge_id = next_ridge
            next_ridge += 1
            candidates.append(
                {
                    "ridge_id": ridge_id,
                    "last_sample": sample_index,
                    "last_energy": float(row["energy"]),
                    "slope": None,
                    "n_points": 1,
                }
            )
        else:
            ridge = candidates[best_idx]
            ridge_id = int(ridge["ridge_id"])
            gap = max(sample_index - int(ridge["last_sample"]), 1)
            new_slope = (float(row["energy"]) - float(ridge["last_energy"])) / gap
            ridge["slope"] = new_slope if ridge["slope"] is None else 0.7 * float(ridge["slope"]) + 0.3 * new_slope
            ridge["last_sample"] = sample_index
            ridge["last_energy"] = float(row["energy"])
            ridge["n_points"] = int(ridge["n_points"]) + 1
        points.at[row_idx, "ridge_id"] = ridge_id

        active[scan_id] = [
            ridge
            for ridge in candidates
            if sample_index - int(ridge["last_sample"]) <= config.max_sample_gap
        ]

    ridge_points = points[points["ridge_id"] >= 0].copy()
    counts = ridge_points.groupby("ridge_id").size()
    keep_ids = counts[counts >= config.min_points].index
    ridge_points = ridge_points[ridge_points["ridge_id"].isin(keep_ids)].copy()
    ridges = summarize_ridges(ridge_points)
    return ridge_points, ridges


def summarize_ridges(ridge_points: pd.DataFrame) -> pd.DataFrame:
    """Summarize linked ridge point table into one row per ridge."""

    if ridge_points.empty:
        return _empty_ridge_table()

    rows: list[dict[str, Any]] = []
    for ridge_id, group in ridge_points.groupby("ridge_id", sort=True):
        group = group.sort_values("sample_index")
        sample_span = int(group["sample_index"].max() - group["sample_index"].min() + 1)
        gap_count = int(sample_span - len(group))
        if len(group) > 1 and np.ptp(group["angle"].to_numpy(float)) > np.finfo(float).eps:
            slope = float(np.polyfit(group["angle"], group["energy"], deg=1)[0])
        else:
            slope = np.nan
        rows.append(
            {
                "ridge_id": int(ridge_id),
                "scan_id": group["scan_id"].iloc[0],
                "n_points": int(len(group)),
                "sample_min": int(group["sample_index"].min()),
                "sample_max": int(group["sample_index"].max()),
                "angle_min": float(group["angle"].min()),
                "angle_max": float(group["angle"].max()),
                "energy_min": float(group["energy"].min()),
                "energy_max": float(group["energy"].max()),
                "integrated_prominence": float(group["prominence"].sum()),
                "median_snr": float(group["snr"].median()) if "snr" in group else np.nan,
                "gap_count": gap_count,
                "slope_dE_dangle": slope,
                "quality_score": float(group["prominence"].sum() / max(1, gap_count + 1)),
            }
        )
    return pd.DataFrame(rows)


def filter_ridge_candidates(
    ridge_points: pd.DataFrame,
    ridges: pd.DataFrame,
    *,
    config: RidgeFilterConfig = RidgeFilterConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter ridge candidates to Bragg-like traces.

    This is the first reduction from permissive image peaks to candidate data
    points worth comparing against manual selections.
    """

    if ridge_points.empty or ridges.empty:
        return ridge_points.copy(), ridges.copy()

    accepted = ridges.copy()
    accepted["angle_span"] = accepted["angle_max"] - accepted["angle_min"]
    accepted["energy_span"] = accepted["energy_max"] - accepted["energy_min"]
    accepted["gap_fraction"] = accepted["gap_count"] / np.maximum(
        accepted["sample_max"] - accepted["sample_min"] + 1,
        1,
    )

    accepted["sample_span"] = accepted["sample_max"] - accepted["sample_min"]

    mask = (
        (accepted["n_points"] >= config.min_points)
        & (accepted["sample_span"] >= config.min_sample_span)
        & (accepted["energy_span"] >= config.min_energy_span)
        & (accepted["gap_fraction"] <= config.max_gap_fraction)
    )
    if config.min_angle_span > 0:
        mask &= accepted["angle_span"] >= config.min_angle_span
    if config.min_abs_slope_dE_dangle is not None:
        slope = accepted["slope_dE_dangle"].abs()
        mask &= slope.notna() & (slope >= config.min_abs_slope_dE_dangle)

    accepted = accepted[mask].copy()

    if config.top_n_ridges is not None and len(accepted) > config.top_n_ridges:
        # Salience = integrated brightness × sample coverage × |slope|.
        # Enforces the physics prior that only O(10) real Bragg lines exist per scan.
        slope_abs = accepted["slope_dE_dangle"].abs().fillna(0.0)
        salience = accepted["integrated_prominence"] * accepted["sample_span"] * slope_abs
        accepted = accepted.loc[salience.nlargest(config.top_n_ridges).index].copy()

    filtered_points = ridge_points[ridge_points["ridge_id"].isin(accepted["ridge_id"])].copy()
    return filtered_points, accepted


def _empty_ridge_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ridge_id",
            "scan_id",
            "n_points",
            "sample_min",
            "sample_max",
            "angle_min",
            "angle_max",
            "energy_min",
            "energy_max",
            "integrated_prominence",
            "median_snr",
            "gap_count",
            "slope_dE_dangle",
            "quality_score",
        ]
    )


def export_manual_selection(
    selected_points_by_group: Mapping[str, Sequence[int]],
    pitches: Sequence[float],
    energies: Sequence[float],
    *,
    roll: float | None = None,
    scan_id: str = "",
    h_to_meas: Sequence[int | None] | None = None,
    hquery: Sequence[Any] | None = None,
    h_to_meas_dyndiff: Sequence[int | None] | None = None,
    hquery_dyndiff: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Export notebook hand-selected point groups into a stable table."""

    pitches = np.asarray(pitches, dtype=float)
    energies = np.asarray(energies, dtype=float)
    h_by_group = _invert_h_mapping(h_to_meas, hquery)
    hd_by_group = _invert_h_mapping(h_to_meas_dyndiff, hquery_dyndiff)

    rows: list[dict[str, Any]] = []
    for group_position, (group_label, point_indices) in enumerate(selected_points_by_group.items()):
        for point_index in np.asarray(point_indices, dtype=int).ravel():
            if point_index < 0 or point_index >= len(pitches) or point_index >= len(energies):
                rows.append(
                    {
                        "scan_id": scan_id,
                        "group_id": group_position,
                        "group_label": group_label,
                        "point_index": int(point_index),
                        "pitch": np.nan,
                        "roll": roll,
                        "energy": np.nan,
                        "manual_H": h_by_group.get(group_position),
                        "manual_H_dyndiff": hd_by_group.get(group_position),
                        "valid": False,
                    }
                )
                continue
            rows.append(
                {
                    "scan_id": scan_id,
                    "group_id": group_position,
                    "group_label": group_label,
                    "point_index": int(point_index),
                    "pitch": float(pitches[point_index]),
                    "roll": roll,
                    "energy": float(energies[point_index]),
                    "manual_H": h_by_group.get(group_position),
                    "manual_H_dyndiff": hd_by_group.get(group_position),
                    "valid": True,
                }
            )
    return pd.DataFrame(rows)


def _invert_h_mapping(
    h_to_meas: Sequence[int | None] | None,
    hquery: Sequence[Any] | None,
) -> dict[int, str]:
    if h_to_meas is None or hquery is None:
        return {}
    mapping: dict[int, str] = {}
    for h_value, group_id in zip(hquery, h_to_meas):
        if group_id is None:
            continue
        try:
            if pd.isna(group_id):
                continue
        except TypeError:
            pass
        mapping[int(group_id)] = str(h_value)
    return mapping


def compare_manual_to_candidates(
    manual_points: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    energy_tolerance: float,
    angle_tolerance: float | None = None,
    manual_angle_column: str = "pitch",
    candidate_angle_column: str = "angle",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Match hand-selected points to nearest automated candidates."""

    if manual_points.empty:
        return pd.DataFrame(), {"manual_count": 0.0, "recovered_fraction": np.nan}
    if candidates.empty:
        result = manual_points.copy()
        result["matched"] = False
        result["candidate_index"] = np.nan
        result["delta_energy"] = np.nan
        result["delta_angle"] = np.nan
        return result, {"manual_count": float(len(result)), "recovered_fraction": 0.0}

    rows: list[dict[str, Any]] = []
    for manual_idx, manual in manual_points.iterrows():
        scan_candidates = candidates
        if "scan_id" in manual_points.columns and "scan_id" in candidates.columns:
            scan_candidates = candidates[candidates["scan_id"] == manual.get("scan_id")]
        if scan_candidates.empty or not bool(manual.get("valid", True)):
            rows.append(_manual_match_row(manual_idx, manual, None, np.nan, np.nan, False))
            continue
        delta_e = scan_candidates["energy"].to_numpy(float) - float(manual["energy"])
        if angle_tolerance is not None:
            delta_a = scan_candidates[candidate_angle_column].to_numpy(float) - float(manual[manual_angle_column])
            scaled = (delta_e / energy_tolerance) ** 2 + (delta_a / angle_tolerance) ** 2
        else:
            delta_a = np.zeros_like(delta_e)
            scaled = (delta_e / energy_tolerance) ** 2
        best_pos = int(np.nanargmin(scaled))
        candidate = scan_candidates.iloc[best_pos]
        matched = abs(delta_e[best_pos]) <= energy_tolerance
        if angle_tolerance is not None:
            matched = matched and abs(delta_a[best_pos]) <= angle_tolerance
        rows.append(
            _manual_match_row(
                manual_idx,
                manual,
                candidate,
                float(delta_e[best_pos]),
                float(delta_a[best_pos]),
                bool(matched),
            )
        )

    result = pd.DataFrame(rows)
    valid = result["valid"].fillna(True)
    metrics = {
        "manual_count": float(valid.sum()),
        "matched_count": float((result["matched"] & valid).sum()),
        "recovered_fraction": float((result["matched"] & valid).sum() / max(valid.sum(), 1)),
        "median_abs_delta_energy": float(result.loc[result["matched"], "delta_energy"].abs().median())
        if result["matched"].any()
        else np.nan,
        "median_abs_delta_angle": float(result.loc[result["matched"], "delta_angle"].abs().median())
        if result["matched"].any()
        else np.nan,
    }
    return result, metrics


def _manual_match_row(
    manual_idx: Any,
    manual: pd.Series,
    candidate: pd.Series | None,
    delta_energy: float,
    delta_angle: float,
    matched: bool,
) -> dict[str, Any]:
    row = manual.to_dict()
    row["manual_row"] = manual_idx
    row["matched"] = matched
    row["candidate_index"] = candidate.name if candidate is not None else np.nan
    row["candidate_ridge_id"] = candidate.get("ridge_id", np.nan) if candidate is not None else np.nan
    row["candidate_energy"] = candidate.get("energy", np.nan) if candidate is not None else np.nan
    row["candidate_angle"] = candidate.get("angle", np.nan) if candidate is not None else np.nan
    row["delta_energy"] = delta_energy
    row["delta_angle"] = delta_angle
    return row


def build_curve_table(
    curves: Iterable[Mapping[str, Any]],
    *,
    seed_id: str = "",
    scan_id: str = "",
) -> pd.DataFrame:
    """Create a normalized curve-point table from notebook-generated curves.

    Each item in `curves` should provide `H`, `angle`, and `energy` arrays.
    Optional keys are copied to every point.
    """

    rows: list[dict[str, Any]] = []
    for curve_id, curve in enumerate(curves):
        angles = np.asarray(curve["angle"], dtype=float)
        energies = np.asarray(curve["energy"], dtype=float)
        if angles.size != energies.size:
            raise ValueError("Curve angle and energy arrays must have the same length")
        h_value = curve.get("H", "")
        base = {k: v for k, v in curve.items() if k not in {"angle", "energy", "H"}}
        for idx, (angle, energy) in enumerate(zip(angles, energies)):
            rows.append(
                {
                    "seed_id": seed_id,
                    "scan_id": scan_id,
                    "curve_id": curve_id,
                    "curve_point_index": idx,
                    "H": str(h_value),
                    "angle": float(angle),
                    "energy": float(energy),
                    **base,
                }
            )
    return pd.DataFrame(rows)


def score_curves_against_points(
    curve_points: pd.DataFrame,
    points: pd.DataFrame,
    *,
    energy_scale: float = 20.0,
    max_energy_residual: float = 80.0,
    intensity_column: str = "prominence",
) -> pd.DataFrame:
    """Score candidate curves against peak/ridge points by interpolated residual."""

    if curve_points.empty or points.empty:
        return pd.DataFrame(
            columns=[
                "seed_id",
                "scan_id",
                "curve_id",
                "H",
                "n_support",
                "score",
                "median_abs_residual",
            ]
        )

    rows: list[dict[str, Any]] = []
    group_cols = ["seed_id", "scan_id", "curve_id", "H"]
    for keys, curve in curve_points.groupby(group_cols, dropna=False):
        seed_id, scan_id, curve_id, h_value = keys
        curve = curve.sort_values("angle")
        unique_curve = curve.drop_duplicates("angle")
        if unique_curve.shape[0] < 2:
            continue
        local_points = points
        if "scan_id" in points.columns and scan_id:
            local_points = points[points["scan_id"] == scan_id]
        if local_points.empty:
            continue
        angle_min = unique_curve["angle"].min()
        angle_max = unique_curve["angle"].max()
        local_points = local_points[
            (local_points["angle"] >= angle_min) & (local_points["angle"] <= angle_max)
        ]
        if local_points.empty:
            continue
        predicted = np.interp(
            local_points["angle"].to_numpy(float),
            unique_curve["angle"].to_numpy(float),
            unique_curve["energy"].to_numpy(float),
        )
        residual = local_points["energy"].to_numpy(float) - predicted
        support = np.abs(residual) <= max_energy_residual
        if not np.any(support):
            continue
        weights = (
            local_points[intensity_column].to_numpy(float)
            if intensity_column in local_points
            else np.ones(local_points.shape[0])
        )
        robust_score = weights[support] * np.exp(-0.5 * (residual[support] / energy_scale) ** 2)
        rows.append(
            {
                "seed_id": seed_id,
                "scan_id": scan_id,
                "curve_id": curve_id,
                "H": h_value,
                "n_support": int(np.sum(support)),
                "score": float(np.sum(robust_score)),
                "median_abs_residual": float(np.median(np.abs(residual[support]))),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "seed_id",
                "scan_id",
                "curve_id",
                "H",
                "n_support",
                "score",
                "median_abs_residual",
            ]
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def soft_assign_points_to_curves(
    curve_points: pd.DataFrame,
    points: pd.DataFrame,
    *,
    energy_scale: float = 20.0,
    max_energy_residual: float = 80.0,
    top_k: int = 3,
    intensity_column: str = "prominence",
) -> pd.DataFrame:
    """Return soft point-to-curve assignment candidates.

    The output keeps multiple plausible H curves per point. It is meant for
    diagnostics and latent-assignment fitting, not for immediate hard labels.
    """

    columns = [
        "point_index",
        "seed_id",
        "scan_id",
        "curve_id",
        "H",
        "point_angle",
        "point_energy",
        "predicted_energy",
        "residual",
        "assignment_score",
        "assignment_weight",
    ]
    if curve_points.empty or points.empty:
        return pd.DataFrame(columns=columns)

    assignment_rows: list[dict[str, Any]] = []
    group_cols = ["seed_id", "scan_id", "curve_id", "H"]
    curves = []
    for keys, curve in curve_points.groupby(group_cols, dropna=False):
        seed_id, scan_id, curve_id, h_value = keys
        curve = curve.sort_values("angle").drop_duplicates("angle")
        if curve.shape[0] < 2:
            continue
        curves.append(
            {
                "seed_id": seed_id,
                "scan_id": scan_id,
                "curve_id": curve_id,
                "H": h_value,
                "angle": curve["angle"].to_numpy(float),
                "energy": curve["energy"].to_numpy(float),
            }
        )

    for point_index, point in points.iterrows():
        candidates = []
        point_scan = point.get("scan_id", "")
        point_angle = float(point["angle"])
        point_energy = float(point["energy"])
        point_weight = float(point.get(intensity_column, 1.0)) if intensity_column in point else 1.0
        for curve in curves:
            if curve["scan_id"] and point_scan and curve["scan_id"] != point_scan:
                continue
            if point_angle < curve["angle"][0] or point_angle > curve["angle"][-1]:
                continue
            predicted = float(np.interp(point_angle, curve["angle"], curve["energy"]))
            residual = point_energy - predicted
            if abs(residual) > max_energy_residual:
                continue
            score = point_weight * np.exp(-0.5 * (residual / energy_scale) ** 2)
            candidates.append(
                {
                    "point_index": point_index,
                    "seed_id": curve["seed_id"],
                    "scan_id": point_scan,
                    "curve_id": curve["curve_id"],
                    "H": curve["H"],
                    "point_angle": point_angle,
                    "point_energy": point_energy,
                    "predicted_energy": predicted,
                    "residual": float(residual),
                    "assignment_score": float(score),
                }
            )
        if not candidates:
            continue
        candidates = sorted(candidates, key=lambda row: row["assignment_score"], reverse=True)[:top_k]
        total = sum(row["assignment_score"] for row in candidates)
        for row in candidates:
            row["assignment_weight"] = row["assignment_score"] / total if total > 0 else np.nan
            assignment_rows.append(row)

    return pd.DataFrame(assignment_rows, columns=columns)


def residual_table(
    assigned_points: pd.DataFrame,
    predictor: Callable[[Any, np.ndarray, np.ndarray], np.ndarray],
    *,
    h_column: str = "H",
    angle_column: str = "angle",
    roll_column: str = "roll",
    energy_column: str = "energy",
) -> pd.DataFrame:
    """Compute residuals for assigned points using a notebook-provided predictor."""

    rows = []
    for h_value, group in assigned_points.groupby(h_column):
        angles = group[angle_column].to_numpy(float)
        rolls = group[roll_column].to_numpy(float) if roll_column in group else np.zeros_like(angles)
        predicted = np.asarray(predictor(h_value, angles, rolls), dtype=float)
        residual = group[energy_column].to_numpy(float) - predicted
        local = group.copy()
        local["predicted_energy"] = predicted
        local["residual"] = residual
        rows.append(local)
    return pd.concat(rows, ignore_index=True) if rows else assigned_points.copy()


def find_global_energy_offset(
    ridge_points: pd.DataFrame,
    curve_points: pd.DataFrame,
    *,
    search_range_eV: float = 200.0,
    energy_scale: float = 15.0,
    n_steps: int = 400,
    intensity_column: str = "prominence",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Find the global energy shift ΔE that best aligns model curves to image ridges.

    Uses a KDE / matched-filter approach: for each ridge point the signed
    residual to its nearest model curve at ΔE=0 is computed.  The KDE of
    those residuals, weighted by ridge-point brightness, peaks at the
    required ΔE.  This is robust to wrong H-plane assignments because each
    point only contributes via its *nearest* curve.

    Parameters
    ----------
    ridge_points:
        Output of ``filter_ridge_candidates`` (has ``angle``, ``energy``,
        ``prominence`` columns).
    curve_points:
        Output of ``build_curve_table`` (has ``angle``, ``energy``, ``H``
        columns).  Evaluated at the current (possibly wrong) parameter guess.
    search_range_eV:
        Half-width of the ΔE search window.
    energy_scale:
        Gaussian bandwidth for the KDE, ≈ expected energy uncertainty per point.
    n_steps:
        Number of ΔE trial values.
    intensity_column:
        Ridge-point column used as KDE weight.

    Returns
    -------
    best_delta_E:
        Scalar offset in eV.
    delta_E_grid:
        Array of trial ΔE values (length ``n_steps``).
    score_grid:
        KDE score at each trial ΔE.
    """
    delta_E_grid = np.linspace(-search_range_eV, search_range_eV, n_steps)

    if ridge_points.empty or curve_points.empty:
        return 0.0, delta_E_grid, np.zeros(n_steps)

    group_cols = ["seed_id", "scan_id", "curve_id", "H"]
    curves: list[dict[str, Any]] = []
    for _, grp in curve_points.groupby(
        [c for c in group_cols if c in curve_points.columns], dropna=False
    ):
        grp = grp.sort_values("angle").drop_duplicates("angle")
        if len(grp) >= 2:
            curves.append(
                {
                    "angle": grp["angle"].to_numpy(float),
                    "energy": grp["energy"].to_numpy(float),
                }
            )

    if not curves:
        return 0.0, delta_E_grid, np.zeros(n_steps)

    nearest_residuals: list[float] = []
    weights_list: list[float] = []

    for _, pt in ridge_points.iterrows():
        a = float(pt["angle"])
        e = float(pt["energy"])
        w = float(pt.get(intensity_column, 1.0)) if intensity_column in pt else 1.0

        r_candidates: list[float] = []
        for c in curves:
            if a < c["angle"][0] or a > c["angle"][-1]:
                continue
            e_model = float(np.interp(a, c["angle"], c["energy"]))
            r_candidates.append(e - e_model)

        if not r_candidates:
            continue
        # Take the residual to the nearest curve (minimum absolute value).
        r_nearest = r_candidates[int(np.argmin(np.abs(r_candidates)))]
        nearest_residuals.append(r_nearest)
        weights_list.append(w)

    if not nearest_residuals:
        return 0.0, delta_E_grid, np.zeros(n_steps)

    r_arr = np.array(nearest_residuals)
    w_arr = np.array(weights_list, dtype=float)
    w_arr /= w_arr.sum()

    # KDE: score(ΔE) = Σ_i w_i · exp(−(r_i − ΔE)² / 2σ²)
    diff = r_arr[:, None] - delta_E_grid[None, :]  # (n_pts, n_steps)
    score_grid = (w_arr[:, None] * np.exp(-0.5 * (diff / energy_scale) ** 2)).sum(axis=0)

    best_delta_E = float(delta_E_grid[int(np.argmax(score_grid))])
    return best_delta_E, delta_E_grid, score_grid


def hard_assign_ridges_to_hplanes(
    filtered_ridges: pd.DataFrame,
    filtered_ridge_points: pd.DataFrame,
    curve_points: pd.DataFrame,
    delta_E: float,
    *,
    max_residual_eV: float = 40.0,
    intensity_column: str = "prominence",
) -> pd.DataFrame:
    """Assign each filtered ridge to its best-matching H-plane given a global ΔE.

    The model energies in ``curve_points`` are shifted by ``delta_E`` before
    matching, so ``delta_E`` should come from ``find_global_energy_offset``.

    Parameters
    ----------
    filtered_ridges:
        Summary table from ``filter_ridge_candidates``.
    filtered_ridge_points:
        Point table from ``filter_ridge_candidates`` (has ``ridge_id``,
        ``angle``, ``energy``, ``prominence``).
    curve_points:
        Bragg curve samples from ``build_curve_table``.
    delta_E:
        Global energy shift (eV) to add to model energies before matching.
    max_residual_eV:
        Ridges with |median_residual| > this after shifting are left unassigned.
    intensity_column:
        Column used for weighting within each ridge.

    Returns
    -------
    DataFrame with columns: ``ridge_id``, ``scan_id``, ``H``, ``n_points``,
    ``median_residual``, ``rms_residual``, ``weight``.
    Only successfully assigned ridges are included.
    """
    if filtered_ridges.empty or filtered_ridge_points.empty or curve_points.empty:
        return pd.DataFrame(
            columns=["ridge_id", "scan_id", "H", "n_points", "median_residual", "rms_residual", "weight"]
        )

    group_cols = ["seed_id", "scan_id", "curve_id", "H"]
    curves: list[dict[str, Any]] = []
    for keys, grp in curve_points.groupby(
        [c for c in group_cols if c in curve_points.columns], dropna=False
    ):
        grp = grp.sort_values("angle").drop_duplicates("angle")
        if len(grp) >= 2:
            h_val = grp["H"].iloc[0] if "H" in grp.columns else ""
            curves.append(
                {
                    "H": h_val,
                    "angle": grp["angle"].to_numpy(float),
                    "energy": grp["energy"].to_numpy(float) + delta_E,
                }
            )

    if not curves:
        return pd.DataFrame(
            columns=["ridge_id", "scan_id", "H", "n_points", "median_residual", "rms_residual", "weight"]
        )

    rows: list[dict[str, Any]] = []
    for _, ridge in filtered_ridges.iterrows():
        rid = int(ridge["ridge_id"])
        scan_id = ridge.get("scan_id", "")
        pts = filtered_ridge_points[filtered_ridge_points["ridge_id"] == rid]
        if pts.empty:
            continue

        best_h: str | None = None
        best_med: float = np.inf
        best_rms: float = np.inf
        best_resids: np.ndarray = np.array([])

        for c in curves:
            a_arr = pts["angle"].to_numpy(float)
            e_arr = pts["energy"].to_numpy(float)
            in_range = (a_arr >= c["angle"][0]) & (a_arr <= c["angle"][-1])
            if in_range.sum() < 2:
                continue
            e_model = np.interp(a_arr[in_range], c["angle"], c["energy"])
            resids = e_arr[in_range] - e_model
            med = float(np.median(np.abs(resids)))
            if med < best_med:
                best_med = med
                best_h = c["H"]
                best_rms = float(np.sqrt(np.mean(resids**2)))
                best_resids = resids

        if best_h is None or best_med > max_residual_eV:
            continue

        w = float(pts[intensity_column].sum()) if intensity_column in pts.columns else float(len(pts))
        rows.append(
            {
                "ridge_id": rid,
                "scan_id": scan_id,
                "H": best_h,
                "n_points": int(len(best_resids)),
                "median_residual": float(np.median(best_resids)),
                "rms_residual": best_rms,
                "weight": w,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["ridge_id", "scan_id", "H", "n_points", "median_residual", "rms_residual", "weight"],
    )


def dispersion_scale_from_ridge_slopes(
    assignments: pd.DataFrame,
    filtered_ridge_points: pd.DataFrame,
    curve_points: pd.DataFrame,
    *,
    ecentral: float = 12407.0,
    min_angle_span: float = 0.05,
    k_bounds: tuple[float, float] = (0.4, 1.6),
) -> dict[str, Any]:
    r"""Estimate the spectrometer dispersion-scale error from Bragg-ridge slopes.

    This implements the dispersion-free comparison trick: the measured energy
    axis is affine in the true energy, ``E_meas - ecentral = k_err * (E_true -
    ecentral)``, where ``k_err = dEdP_assumed / dEdP_true`` is the unknown
    dispersion error.  A ridge's slope scales the same way, so the ratio of a
    model curve's slope to the measured ridge slope recovers the correction
    factor ``k = slope_model / slope_meas = 1 / k_err``.  The corrected axis is
    then ``E_corr = ecentral + k * (E_meas - ecentral)`` — no hand-set ``dEdP``
    needed.

    The estimate is pooled over all confidently assigned ridges, weighted by
    ``angle_span**2`` (longer ridges give a better slope lever arm).

    .. warning::
       This is only well-determined when at least one assigned ridge has real
       angular extent.  On the oct2024 *pitch* scans the Bragg curves sit near
       their turning point and span only ~0.03-0.07 deg, so individual slopes
       are ill-conditioned and the pooled estimate is unreliable (verified:
       k scattered 0.67-1.08 vs the true 0.70).  For such data, fit the
       dispersion scale jointly across multiple scans at different central
       energies instead, and use this function only as a cross-check or for
       wide-angle (e.g. roll) scans.

    Parameters
    ----------
    assignments:
        Output of :func:`hard_assign_ridges_to_hplanes` (columns ``ridge_id``,
        ``H``).
    filtered_ridge_points:
        Point table from :func:`filter_ridge_candidates` (``ridge_id``,
        ``angle``, ``energy``).  Energies must be on the *measured* axis whose
        scale you want to correct.
    curve_points:
        Bragg-curve samples from :func:`build_curve_table`, in true energy.
    ecentral:
        Pivot energy of the affine axis transform.
    min_angle_span:
        Ridges narrower than this (deg) are skipped — their slope is unreliable.
    k_bounds:
        Sanity bounds on the per-ridge ratio; outliers are discarded.

    Returns
    -------
    dict with keys:
        ``k`` (span-weighted-mean correction, or ``nan`` if no usable ridge),
        ``k_median``, ``n_ridges`` (number contributing), ``per_ridge``
        (list of ``(H, angle_span, slope_meas, slope_model, k)`` tuples).
    """
    result: dict[str, Any] = {"k": np.nan, "k_median": np.nan, "n_ridges": 0, "per_ridge": []}
    if assignments.empty or filtered_ridge_points.empty or curve_points.empty:
        return result

    group_cols = [c for c in ["seed_id", "scan_id", "curve_id", "H"] if c in curve_points.columns]
    curves: dict[str, dict[str, np.ndarray]] = {}
    for _, grp in curve_points.groupby(group_cols, dropna=False):
        grp = grp.sort_values("angle").drop_duplicates("angle")
        if len(grp) >= 2 and "H" in grp.columns:
            curves[str(grp["H"].iloc[0])] = {
                "angle": grp["angle"].to_numpy(float),
                "energy": grp["energy"].to_numpy(float),
            }

    ks: list[float] = []
    weights: list[float] = []
    for _, asn in assignments.iterrows():
        rid = int(asn["ridge_id"])
        h_key = str(asn["H"])
        if h_key not in curves:
            continue
        pts = filtered_ridge_points[filtered_ridge_points["ridge_id"] == rid].sort_values("angle")
        if len(pts) < 3:
            continue
        angles = pts["angle"].to_numpy(float)
        span = float(angles.max() - angles.min())
        if span < min_angle_span or np.ptp(angles) < 1e-9:
            continue
        energies = pts["energy"].to_numpy(float)
        slope_meas = float(np.polyfit(angles, energies, 1)[0])
        if abs(slope_meas) < np.finfo(float).eps:
            continue
        c = curves[h_key]
        i0 = int(np.argmin(np.abs(c["angle"] - angles.mean())))
        slope_model = float(np.gradient(c["energy"], c["angle"])[i0])
        k = slope_model / slope_meas
        if not (k_bounds[0] < k < k_bounds[1]):
            continue
        ks.append(k)
        weights.append(span**2)
        result["per_ridge"].append((asn["H"], span, slope_meas, slope_model, k))

    if not ks:
        return result
    ks_arr = np.array(ks)
    w_arr = np.array(weights)
    result["k"] = float(np.average(ks_arr, weights=w_arr))
    result["k_median"] = float(np.median(ks_arr))
    result["n_ridges"] = len(ks_arr)
    return result
