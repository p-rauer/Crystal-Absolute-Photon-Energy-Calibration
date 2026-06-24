"""TEMPLATE campaign driver for the crystal-calibration pipeline.

Copy this into a per-campaign folder (e.g. `myrun/calibrate_myrun.py`) and fill in the
four campaign-specific things below. The reusable engine is functions/calibration_pipeline.py;
the full rationale + pitfalls are in Calibration_Recipe.md. The apr2026/ folder is a worked
example.

Per-campaign work is ONLY:
  1. load_scan(fn)  - read one scan into (image, energy_axis_raw, angles, fixed_angle, axis)
  2. SCAN_FILES     - a pitch-diverse set (need both low- and high-pitch scans for the tilt)
  3. ref_plane      - a long, SHALLOW line (pins dispersion + Ecorr; e.g. [-1,-1,5] at 9 keV)
  4. anchors / mergers - a few HAND-IDENTIFIED planes for the steep near-vertical lines and
                         any h<->k merger (auto best-plane matching CANNOT be trusted for the
                         steep lines - they sit ~200 eV from their true plane at a wrong geom)

Run:  python calibrate_myrun.py            # full seedless pipeline
      python calibrate_myrun.py --quick    # refine from --seed (or a hard-coded guess)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import sys

import numpy as np

# repo root = parent of this campaign folder (adjust if you place the driver elsewhere)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.calibration_pipeline import (
    Campaign, PlaneAnchor, LineMerger, run, extract, estimate_disp, refine,
    centroid_ecorr, validate, plot_overlay)

# --- 1. campaign data location -------------------------------------------------------
D = "/path/to/CalibrationData/MYRUN"            # TODO
EREGION = (9035.0, 9115.0)                       # TODO interesting (uncorrected) energy band


def load_scan(fn):
    """TODO: return (image[n_angle, n_energy], energy_raw[n_energy], angles[n_angle],
    fixed_angle_deg, axis) where axis is 'pitch' (roll fixed) or 'roll' (pitch fixed).

    Watch out for (see Calibration_Recipe.md):
      * energy zero-point may be TIME-DEPENDENT (a step at some timestamp) - put it here.
      * angle = motor + mechanical offset; the non-scanned angle is read from the status file.
      * sort by the scanned angle; downsample very long scans (>~2500 pts) for speed.
    """
    raise NotImplementedError("fill in load_scan for this campaign")


# --- 2. pitch-diverse scan set (LOW + HIGH pitch needed to constrain the axis tilt) --
SCAN_FILES = [
    # "...high_pitch_scan_with_the_long_shallow_ref_line...",
    # "...low_pitch_scan_with_the_steep_lines...",
]                                                # TODO

# --- 3 & 4. physics inputs -----------------------------------------------------------
STEEP = (-1e9, -150.0)                            # raw-axis dE/dpitch of near-vertical lines

CAMPAIGN = Campaign(
    name="myrun",                                # TODO
    scan_files=SCAN_FILES,
    load=load_scan,
    prior=(1.32, 0.0, 0.0),                      # (dP, dR, dY): dP from the motor offset;
                                                 # dR/dY = 0 (the seedless grid finds them)
    ref_plane=(-1, -1, 5),                       # TODO long shallow line (disp + Ecorr anchor)
    ref_scans=None,                              # scans carrying it (default: all)
    anchors=[
        # TODO: identify the steep near-vertical lines by eye and pin them here. Window is
        # (pitch_lo, pitch_hi) deg and (slope_lo, slope_hi) eV/deg on the RAW energy axis.
        # PlaneAnchor(pitch=(34.3, 35.5), slope=STEEP, H=(-1, 1, 1)),
    ],
    mergers=[
        # TODO: any h<->k pair that visibly MERGES into one ridge (effective roll ~ 0 there).
        # LineMerger(pitch=(65.4, 66.8), H1=(-3, -1, 1), H2=(-1, -3, 1), weight=8.0),
    ],
    # disp confined to a +-disp_cone band around the ref-line slope-ratio (not hard-fixed):
    fix_disp=True, disp_cone=0.02,
    # anchor_weight=3 is the balanced default; raising it pulls the steep anchors harder but
    # can trade against other anchors (a genuine 2-pitch-group limitation). pin_dY only if a
    # flat dY valley remains AND you have read dY off the FULL steep lines by eye.
    anchor_weight=3.0, pin_dY=None,
)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the grid; refine from --seed")
    ap.add_argument("--seed", type=float, nargs=7, default=None,
                    metavar=("dP", "dR", "dY", "dRP", "dYP", "Ecorr", "disp"))
    args = ap.parse_args()
    HERE = os.path.dirname(os.path.abspath(__file__))

    if args.quick:
        sets = extract(CAMPAIGN)
        disp, pivot, ec0 = estimate_disp(CAMPAIGN, sets)
        p0 = args.seed or [CAMPAIGN.prior[0], 0.5, 0.0, 0.02, 0.0, ec0, disp]
        x, c = refine(CAMPAIGN, sets, p0, pivot, maxiter=400)
        x = centroid_ecorr(CAMPAIGN, sets, x, pivot)
        result = validate(CAMPAIGN, sets, x, pivot)
        plot_overlay(CAMPAIGN, x, pivot, SCAN_FILES[0], os.path.join(HERE, "overlay_high_pitch.png"))
        plot_overlay(CAMPAIGN, x, pivot, SCAN_FILES[-1], os.path.join(HERE, "overlay_low_pitch.png"))
    else:
        result = run(CAMPAIGN, n_refine_seeds=3, out_json="calibration_result.json",
                     overlay_scans=[SCAN_FILES[0], SCAN_FILES[-1]], out_dir=HERE)
    p = result["params"]
    print(f"\nFINAL {CAMPAIGN.name}: dP={p['dP']:.3f} dR={p['dR']:.3f} dY={p['dY']:.3f} "
          f"dRP={p['dRP']:+.4f} dYP={p['dYP']:+.4f} Ecorr={p['Ecorr']:+.2f} "
          f"disp={p['disp']:.4f} (pivot {p['pivot']:.1f})")
    print(f"pass={result['passf']:.2f} median={result['median']:.2f} eV")
