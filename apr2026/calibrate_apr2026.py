"""apr2026 campaign driver for the reusable calibration pipeline.

All campaign-specific knowledge lives HERE; the recipe itself is
functions/calibration_pipeline.py (see Calibration_Recipe.md).

Hand-picked physics inputs (user-identified, 2026-06):
  anchors  - near-vertical lines, which auto-assignment CANNOT be trusted with
             (at a wrong geometry they sit ~200 eV from their true plane):
               pitch ~35  steep   -> [-1, 1,  1]
               pitch ~38  steep   -> [-1, 1, -1]
               pitch ~38.6 shallow-> [-4,-2,  2]
               pitch ~75.2 steep  -> [-2, 0, -2]   (outermost-left of the 75-76 cluster)
  merger   - [-3,-1,1] and [-1,-3,1] merge into one ridge at pitch ~66.1 (h<->k
             degeneracy; soft constraint - depends on roll AND yaw).

Seedlessness: prior is (dP from motor-offset knowledge, dR=0, dY=0) - NOT a previous
fit - the grid stage finds the basin (true answer: dR=0.58, dY=-0.10).

Run:  python calibrate_apr2026.py            # full pipeline (grid stage ~5 min,
                                             # refines ~20 min each on one core)
      python calibrate_apr2026.py --quick    # skip the grid, refine from --seed/prior
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import datetime as dt
import sys

import numpy as np

# repo root is the parent of this campaign folder (apr2026/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.calibration_pipeline import (
    Campaign, PlaneAnchor, LineMerger, run, extract, estimate_disp, refine,
    centroid_ecorr, validate, plot_overlay)

D = "/home/patrickt/DESYCLOUD/PostDoc/HXRSS/CalibrationData/apr2026"
EREGION = (9035.0, 9115.0)               # interesting energy region (uncorrected axis)
# spec_hist energy zero-point is TIME-DEPENDENT (consistency with corr2d):
_OFFSET_CUTOFF = dt.datetime(2026, 4, 10, 11, 17, 0)
# 09_26_27's stated central energy (9080) is a spectrometer-GUI artifact: it is shifted +15 eV
# from the later datasets, NOT a physical change. Subtract 15 -> 9065 to make it comparable to
# the others. (This 15 eV, on the moderate -213 eV/deg [-1,-1,-3] line, was masquerading as a
# 0.07 deg "pitch" shift; on the steep lines it's invisible.) This is a physically-grounded GUI
# correction, not an arbitrary per-scan fudge.
_SCAN_OFFSET = {"20260410-09_26_27": 9065.0}

STEEP = (-1e9, -150.0)                   # raw-axis dE/dpitch of the near-vertical lines


def _energy_offset(fn):
    for k, v in _SCAN_OFFSET.items():
        if k in fn:
            return v
    ts = dt.datetime.strptime(fn[:17], "%Y%m%d-%H_%M_%S")
    return 9080.0 if ts < _OFFSET_CUTOFF else 9065.0


def _roll_angle(motor):
    """roll motor -> deg. (Sign verified against data; an early helper had `-motor`.)"""
    return np.rad2deg(np.arcsin((2.2774 + np.asarray(motor, float)) / 87))


def _status_motor(fn, which):
    """ACTUALPOSITION of the NON-scanned motor from the per-scan status file."""
    fp = f"{D}/{fn}"
    lbl = str(np.load(fp, allow_pickle=True)["doocs_channel"])
    mono = "2252" if "2252" in lbl else "2307"
    addr = f"XFEL.FEL/HXRSS_MONO2_MOTOR.SA2/{which}.{mono}.SA2/ACTUALPOSITION"
    fd = np.loadtxt(fp + "_status.txt", dtype="str", delimiter=",", skiprows=1)
    return float(fd[np.where(fd == addr)[0][0]][1])


def load_scan(fn):
    """(image[n_angle, n_energy], energy_raw, angles, fixed_angle, axis).
    spec_hist + doocs_vals_hist (per-sample - corr2d is pitch-FILTERED and coarser).
    Axis from doocs_channel: MONOPA -> pitch scan (fixed roll from status);
    MONORA -> roll scan (fixed pitch from status). pitch = motor + 259.77."""
    tt = np.load(f"{D}/{fn}", allow_pickle=True)
    S = np.asarray(tt["spec_hist"], float)
    phen = (np.arange(S.shape[1]) - 1550) * 0.18 + _energy_offset(fn)
    m = (phen >= EREGION[0]) & (phen <= EREGION[1])
    S, phen = S[:, m], phen[m]
    motor = np.asarray(tt["doocs_vals_hist"], float)
    if "MONORA" in str(tt["doocs_channel"]):
        ang = _roll_angle(motor); axis = "roll"
        fixed = _status_motor(fn, "MONOPA") + 259.77
    else:
        ang = motor + 259.77; axis = "pitch"
        fixed = float(_roll_angle(_status_motor(fn, "MONORA")))
    o = np.argsort(ang)
    S, ang = S[o], ang[o]
    if len(ang) > 2500:
        k = int(np.ceil(len(ang) / 2500))
        S, ang = S[::k], ang[::k]
    return S, phen, ang, fixed, axis


HIGH_PITCH = [
    "20260410-13_07_49_cor2d.npz",       # 66-77 deg (long [-1,-1,5] diagonal - disp ref)
    "20260410-12_50_02_cor2d.npz",       # 70-77
    "20260410-11_37_47_cor2d.npz",       # 73-76
]
SCAN_FILES = HIGH_PITCH + [
    "20260410-13_47_14_cor2d.npz",       # 34-39 (steep lines: tilt + anchor leverage)
    "20260410-13_54_52_cor2d.npz",       # 34-41
    # DIFFERENT-ROLL pitch scan (roll 0.18 deg vs ~1.372 for all the others): a ~1.2 deg
    # roll offset gives the independent dR/dY leverage the same-roll scans lack - this is
    # what finally breaks the dY<->dR degeneracy from data instead of by eye.
    "20260410-09_26_27_cor2d.npz",       # 65-69 deg @ roll 0.18
]
# ROLL scans (loader supports them; axis auto-detected from doocs_channel). NOT in the
# default fit: (a) their 0.06 deg span at pitch 66.04 does not discriminate the known
# dY<->dYP<->disp degeneracy (both basins predict the X-crossing of [-3,-1,1]/[-1,-3,1]
# at the same roll within 1 mdeg); (b) they fragment into ~40 ridges each, which would
# dominate the equal-weight cost without de-duplication; (c) they are pre-11:17 and the
# model-implied crossing energy suggests the 9080 offset (or the status pitch) may be
# ~15 eV / ~0.1 deg off - unresolved. For FUTURE campaigns: a roll scan spanning
# >+-0.3 deg, ideally at two pitches, is what actually breaks the degeneracy.
ROLL_SCANS = [
    "20260410-10_51_18_cor2d.npz",       # roll 1.34-1.40 @ pitch 66.04
    "20260410-10_52_03_cor2d.npz",       # roll 1.34-1.40 @ pitch 66.04 (repeat)
]

CAMPAIGN = Campaign(
    name="apr2026",
    scan_files=SCAN_FILES,
    load=load_scan,
    prior=(1.32, 0.0, 0.0),              # dP from pitch = motor + 259.77 + mounting prior;
                                         # dR/dY deliberately 0: the grid finds the basin
    # widen the seedless grid: roll/yaw mount offsets up to ~2 deg are physically reasonable
    # (consistent with past campaigns), so dR/dY must be searched that far.
    grid_dR=np.arange(-2.0, 4.01, 0.5),
    grid_dY=np.arange(-2.0, 3.01, 0.5),
    ref_plane=(-1, -1, 5),
    ref_scans=HIGH_PITCH,
    anchors=[
        PlaneAnchor(pitch=(34.3, 35.5), slope=STEEP, H=(-1, 1, 1)),
        PlaneAnchor(pitch=(36.45, 36.85), slope=(300, 600), H=(-1, -1, -1)),
        PlaneAnchor(pitch=(37.5, 38.4), slope=STEEP, H=(-1, 1, -1)),
        PlaneAnchor(pitch=(38.3, 39.3), slope=(-30, 120), H=(-4, -2, 2)),
        PlaneAnchor(pitch=(74.9, 75.35), slope=STEEP, H=(-2, 0, -2)),
        # NOTE all anchor windows are scan-restricted where another scan overlaps the same
        # (pitch, slope) box at a different roll - otherwise the 0.18-deg-roll scan (65-69 deg)
        # would be mis-tagged (its 65-67 ridges are NOT the 66.1 merger, which is roll-1.372).
        # The low (34-39) and 75-76 anchors don't overlap that scan's pitch range, so they
        # need no restriction.
        # basin discriminators (user-confirmed by following the FULL visible lines, not
        # just the extracted ridge fragments): these identities kill the rival basin
        # (dY=-1.4, dYP~+0.006, disp~0.94) that is otherwise nearly cost-degenerate.
        PlaneAnchor(pitch=(75.55, 75.82), slope=(-300, -150), H=(0, -2, -2)),
        # slope window deliberately tight: the +92/+98 ridges at 75.58 are the
        # [-1,-3,3]-family line, NOT [0,-2,-2] (they fail at 12 eV if mis-anchored)
        PlaneAnchor(pitch=(75.55, 75.82), slope=(15, 70), H=(0, -2, -2)),
        PlaneAnchor(pitch=(76.05, 76.30), slope=(-320, -180), H=(-1, -3, 3)),
        # different-roll scan (roll 0.18): the [-3,-1,1]/[-1,-3,1] pair, MERGED at roll 1.37
        # (the merger above), appears SEPARATED here (66.8 and 65.4 deg). Anchoring them in
        # this scan pins the roll-dependence of dR/dY - the leverage the same-roll scans lack.
        # scan-restricted so they don't capture the other scans' ridges.
        # near-vertical fragments span slope +-80..12000 (both signs) -> two wide windows;
        # shallow lines (|slope|<80) are excluded so the ref line isn't captured here.
        # 66.8 steep -> [-3,-1,1], 65.4 -> [-1,-3,1]. This (NO-swap) assignment is correct:
        # swapping them sent the fit to cost ~45 (dP=1.9, disp=1.09 - diverged), vs cost 2.2
        # here. (The h<->k pair's identity at this roll/yaw is not a priori unique; the fit
        # quality settles it decisively in favour of no-swap.)
        PlaneAnchor(pitch=(66.55, 66.95), slope=(80, 1e9), H=(-3, -1, 1), scan="20260410-09_26_27"),
        PlaneAnchor(pitch=(66.55, 66.95), slope=(-1e9, -80), H=(-3, -1, 1), scan="20260410-09_26_27"),
        PlaneAnchor(pitch=(65.20, 65.60), slope=(-120, 200), H=(-1, -3, 1), scan="20260410-09_26_27"),
        # the moderate-slope [-1,-1,-3] line at ~66.16: a real ridge the auto-matcher left
        # unanchored and the detector half-latched onto a bright outlier, hiding a ~0.07 deg
        # pitch discrepancy. Anchoring it forces the GLOBAL geometry to place it correctly.
        PlaneAnchor(pitch=(66.00, 66.32), slope=(-330, -120), H=(-1, -1, -3), scan="20260410-09_26_27"),
    ],
    mergers=[
        LineMerger(pitch=(65.4, 66.8), H1=(-3, -1, 1), H2=(-1, -3, 1), weight=4.0,
                   scan="20260410-13_07_49"),    # roll 1.372 - the only scan where this merges
    ],
    # disp left FULLY OPEN (fix_disp=False): the free optimum (0.862) sits inside the +-2%
    # [-1,-1,5] cone anyway, so the cone wasn't needed for this campaign and open disp fits
    # marginally better (median 0.37 vs 0.45). The grid still relocks disp per node internally.
    # dY is NOT pinned: with 09_26_27 at the GUI-corrected 9065 energy, the fit determines it
    # from the data (dY=+1.0, a physically reasonable ~1 deg mount yaw). fix_disp=True (the
    # +-2% cone) remains available as a safety net for future campaigns where disp runs away.
    fix_disp=False,
    pin_dY=None,
)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the grid; refine from --seed (or the known solution)")
    ap.add_argument("--seed", type=float, nargs=7, default=None,
                    metavar=("dP", "dR", "dY", "dRP", "dYP", "Ecorr", "disp"))
    args = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    if args.quick:
        camp = CAMPAIGN
        sets = extract(camp)
        disp, pivot, ec0 = estimate_disp(camp, sets)
        print(f"disp={disp:.3f} pivot={pivot:.1f} Ecorr~{ec0:+.1f}")
        p0 = args.seed or [1.342, 0.576, -0.099, 0.0277, -0.0001, ec0, disp]
        x, c = refine(camp, sets, p0, pivot, maxiter=400)
        x = centroid_ecorr(camp, sets, x, pivot)
        result = validate(camp, sets, x, pivot)
        plot_overlay(camp, x, pivot, SCAN_FILES[0], os.path.join(HERE, "overlay_high_pitch.png"))
        plot_overlay(camp, x, pivot, SCAN_FILES[-1], os.path.join(HERE, "overlay_low_pitch.png"))
    else:
        result = run(CAMPAIGN, n_refine_seeds=3, out_json="calibration_result.json",
                     overlay_scans=["20260410-13_07_49_cor2d.npz",   # high pitch
                                    "20260410-13_54_52_cor2d.npz",   # low pitch
                                    "20260410-09_26_27_cor2d.npz"],  # different roll
                     out_dir=HERE)
    p = result["params"]
    print(f"\nFINAL {CAMPAIGN.name}: dP={p['dP']:.3f} dR={p['dR']:.3f} dY={p['dY']:.3f} "
          f"dRP={p['dRP']:+.4f} dYP={p['dYP']:+.4f} Ecorr={p['Ecorr']:+.2f} "
          f"disp={p['disp']:.4f} (pivot {p['pivot']:.1f})")
    print(f"pass={result['passf']:.2f} median={result['median']:.2f} eV")
