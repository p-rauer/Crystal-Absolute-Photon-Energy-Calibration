import numpy as np
import pandas as pd

from functions.calibration_automation import (
    PeakDetectionConfig,
    RidgeFilterConfig,
    RidgeLinkConfig,
    compare_manual_to_candidates,
    detect_peak_candidates,
    export_manual_selection,
    filter_ridge_candidates,
    link_ridge_candidates,
    score_curves_against_points,
    soft_assign_points_to_curves,
)


def _synthetic_spectra():
    rng = np.random.default_rng(7)
    n_samples = 30
    n_energy = 240
    energy = np.linspace(12000.0, 12120.0, n_energy)
    angles = np.linspace(-0.5, 0.5, n_samples)
    spectra = rng.normal(0.0, 0.35, size=(n_samples, n_energy))
    true_energy = 12040.0 + 35.0 * angles
    for idx, center in enumerate(true_energy):
        spectra[idx] += 12.0 * np.exp(-0.5 * ((energy - center) / 1.2) ** 2)
    return spectra, energy, angles, true_energy


def test_peak_detection_and_ridge_linking_recover_synthetic_line():
    spectra, energy, angles, true_energy = _synthetic_spectra()
    peaks = detect_peak_candidates(
        spectra,
        energy,
        angles,
        scan_id="scan",
        config=PeakDetectionConfig(
            background_window=31,
            prominence_sigma=4.0,
            max_peaks_per_spectrum=2,
        ),
    )

    assert len(peaks) >= 25
    nearest = []
    for sample_index, expected in enumerate(true_energy):
        local = peaks[peaks["sample_index"] == sample_index]
        if not local.empty:
            nearest.append(float((local["energy"] - expected).abs().min()))
    assert np.median(nearest) < 1.0

    ridge_points, ridges = link_ridge_candidates(
        peaks,
        config=RidgeLinkConfig(max_sample_gap=2, max_energy_gap=4.0, min_points=10),
    )

    assert not ridge_points.empty
    assert ridges["n_points"].max() >= 20

    filtered_points, filtered_ridges = filter_ridge_candidates(
        ridge_points,
        ridges,
        config=RidgeFilterConfig(
            min_points=10,
            min_angle_span=0.5,
            min_energy_span=10.0,
            min_abs_slope_dE_dangle=10.0,
        ),
    )
    assert not filtered_points.empty
    assert filtered_ridges["n_points"].max() >= 20


def test_manual_selection_export_and_candidate_comparison():
    pitches = np.array([0.0, 0.1, 0.2, 0.3])
    energies = np.array([12000.0, 12005.0, 12010.0, 12015.0])
    manual = export_manual_selection(
        {"Group 1": [1, 2]},
        pitches,
        energies,
        roll=1.5,
        scan_id="scan",
        h_to_meas=[0, None],
        hquery=["[1, 1, 1]", "[2, 2, 0]"],
    )

    assert manual.shape[0] == 2
    assert manual["manual_H"].iloc[0] == "[1, 1, 1]"

    candidates = pd.DataFrame(
        {
            "scan_id": ["scan", "scan"],
            "angle": [0.1, 0.2],
            "energy": [12005.2, 12009.7],
            "prominence": [10.0, 8.0],
        }
    )
    matches, metrics = compare_manual_to_candidates(
        manual,
        candidates,
        energy_tolerance=1.0,
        angle_tolerance=0.02,
    )

    assert matches["matched"].all()
    assert metrics["recovered_fraction"] == 1.0


def test_curve_scoring_prefers_matching_curve():
    points = pd.DataFrame(
        {
            "scan_id": ["scan"] * 5,
            "angle": np.linspace(0.0, 1.0, 5),
            "energy": 100.0 + np.linspace(0.0, 1.0, 5) * 10.0,
            "prominence": np.ones(5),
        }
    )
    curve_points = pd.DataFrame(
        {
            "seed_id": ["seed"] * 10,
            "scan_id": ["scan"] * 10,
            "curve_id": [0] * 5 + [1] * 5,
            "H": ["good"] * 5 + ["bad"] * 5,
            "angle": list(np.linspace(0.0, 1.0, 5)) * 2,
            "energy": list(100.0 + np.linspace(0.0, 1.0, 5) * 10.0)
            + list(150.0 + np.linspace(0.0, 1.0, 5) * 10.0),
        }
    )
    scores = score_curves_against_points(curve_points, points, max_energy_residual=20.0)

    assert scores.iloc[0]["H"] == "good"
    assert scores.iloc[0]["n_support"] == 5

    assignments = soft_assign_points_to_curves(
        curve_points,
        points,
        max_energy_residual=20.0,
        top_k=2,
    )

    assert not assignments.empty
    assert set(assignments["H"]) == {"good"}
    assert np.allclose(assignments["assignment_weight"], 1.0)
