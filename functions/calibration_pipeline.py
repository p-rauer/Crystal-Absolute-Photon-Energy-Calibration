"""Reusable crystal-calibration pipeline (consolidated apr2026 recipe).

Stages (see Calibration_Recipe.md for the full rationale):
  1. extract      - ridge extraction per scan (brightness gate; KEEPS shallow lines)
  2. estimate     - dispersion + Ecorr pre-estimate from a hand-named reference plane
  3. grid         - seedless coarse grid over (dR, dY, dRP, dYP); Ecorr re-optimized per node
  4. refine       - Nelder-Mead on all 7 params (explicit simplex steps - see PITFALL below)
  5. centroid     - Ecorr polish from brightness-centroid traces (geometry FIXED)
  6. validate     - per-ridge report, pass fraction, merger check, labeled overlays

Hand-picked physics inputs (the ONLY per-campaign manual knowledge, both declarative):
  * PlaneAnchor  - "the ridge in this (pitch, slope) window IS plane H". Needed for steep
    near-vertical lines: at a slightly wrong geometry they sit ~200 eV from their true
    plane, so auto best-plane matching silently picks a wrong plane and reports a SMALL
    residual - hiding exactly the error the anchor is meant to expose.
  * LineMerger   - "the two planes H1/H2 merge into ONE ridge in this window". Enforced
    SOFTLY (finite spectrometer/ridge resolution) by pulling BOTH planes onto the ridge.
    Depends on roll AND yaw - do not interpret as a pure roll=0 probe.

Objective convention (October convention): every ridge carries EQUAL weight; the cost is
the mean over ridges of min(median |E_data - E_model|, cap). Not brightness-weighted, not
per-scan-median (a dense high-pitch cluster would dominate the sparse low-pitch scans).

PITFALL 1 (cost us a constant -1 eV bias): scipy Nelder-Mead's default initial simplex
steps a 0.0-valued coordinate by 0.00025 only - a parameter started at 0.0 (e.g. Ecorr)
effectively NEVER MOVES. Always pass an explicit initial_simplex (refine() does).

PITFALL 2 (cost us days): the model offered TWO offset mechanisms - delta_*_initial (a
fixed mount pre-rotation, what render_pitch uses) and delta_* (folded into the measurement
rotation). They disagree by up to ~23 eV for yaw-sensitive (l=+-1) planes once dRP/dYP!=0.
EVERYTHING here (bragg_energy, _E_line, _emat, render_pitch, plot_overlay) now uses
delta_*_initial so the fit and the plots evaluate identical physics.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from functions.calibration_model import (
    _bm, bragg_energy, h_to_array, render_pitch, HMAX_DEFAULT)
from functions.calibration_automation import (
    detect_peak_candidates, link_ridge_candidates, filter_ridge_candidates,
    RidgeFilterConfig)


# A scan is either a PITCH scan (roll fixed) or a ROLL scan (pitch fixed). Roll scans
# constrain dR/dY directly (not via the pitch-axis coupling) and are the cleanest way to
# break the dY <-> dYP <-> disp near-degeneracy of pitch-only data. The loader signals the
# axis by returning a 5th element 'pitch'/'roll' (4-tuple = pitch scan).


def _E_line(H, angles, fixed, axis, geom5):
    """Model E for one plane along the scan axis ('pitch': angles=pitch, fixed=roll;
    'roll': angles=roll, fixed=pitch). delta_*_INITIAL convention (mount pre-rotation) -
    bit-identical to render_pitch / bragg_energy; this is what plot_overlay uses too, so the
    fit and the plot evaluate identical physics. See bragg_energy docstring for why."""
    H = h_to_array(H)
    angles = np.atleast_1d(np.asarray(angles, float))
    out = np.empty(angles.size)
    c = _bm.cryst
    _bm.autoUpdate = False
    try:
        c.delta_pitch = 0.0; c.delta_roll = 0.0; c.delta_yaw = 0.0
        c.delta_pitch_initial = np.deg2rad(-geom5[0])
        c.delta_roll_initial = np.deg2rad(-geom5[1])
        c.delta_yaw_initial = np.deg2rad(-geom5[2])
        pa = np.array([1.0, -geom5[3], -geom5[4]])
        c.pitch_axes = pa / np.linalg.norm(pa)
        for i, v in enumerate(angles):
            if axis == "pitch":
                c.pitch, c.roll = np.deg2rad(v), np.deg2rad(fixed)
            else:
                c.pitch, c.roll = np.deg2rad(fixed), np.deg2rad(v)
            c.yaw = 0.0
            c.calc_RLab2Cryst()
            try:
                out[i] = _bm.bragg_wavelength(exact=False, H=H)[1]
            except Exception:
                out[i] = np.nan
    finally:
        _bm.autoUpdate = True
        c.delta_pitch_initial = 0.0; c.delta_roll_initial = 0.0; c.delta_yaw_initial = 0.0
    return out


def _render(geom5, axis, fixed, ea, angle_grid, hmax):
    """Candidate-plane curves along the scan axis."""
    from functions.calibration_model import render_roll
    if axis == "pitch":
        return render_pitch(geom5, fixed, ea, angle_grid, hmax=hmax)
    return render_roll(geom5, fixed, ea, angle_grid, hmax=hmax)


# ----------------------------------------------------------------------------- config --

@dataclass
class PlaneAnchor:
    """Hard plane assignment: a ridge with median pitch in `pitch` and dE/dpitch (raw,
    uncorrected) in `slope` is plane `H`. `scan` (substring of file name) restricts the
    match; None = any scan."""
    pitch: tuple          # (lo, hi) deg
    slope: tuple          # (lo, hi) eV/deg on the RAW energy axis
    H: tuple              # (h, k, l)
    scan: str = None

    def matches(self, fn, pm, sl):
        if self.scan is not None and self.scan not in fn:
            return False
        return self.pitch[0] <= pm <= self.pitch[1] and self.slope[0] <= sl <= self.slope[1]


@dataclass
class LineMerger:
    """Two planes merged into ONE ridge: both H1 and H2 must lie on the ridge found in the
    (pitch, slope) window. Soft constraint with weight `weight` (in units of one ridge)."""
    pitch: tuple
    H1: tuple
    H2: tuple
    slope: tuple = (-1e9, 1e9)
    scan: str = None
    weight: float = 2.0

    def matches(self, fn, pm, sl):
        if self.scan is not None and self.scan not in fn:
            return False
        return self.pitch[0] <= pm <= self.pitch[1] and self.slope[0] <= sl <= self.slope[1]


@dataclass
class Campaign:
    """Everything campaign-specific. `load(fn)` must return
    (image[n_pitch, n_energy], energy_axis_raw[n_energy], pitch[n_pitch], roll_deg)."""
    name: str
    scan_files: list
    load: callable
    prior: tuple                          # rough (dP, dR, dY); dP from the motor offset
    ref_plane: tuple                      # plane of the long shallow reference line
    ref_scans: list = None                # scans carrying it (default: all)
    anchors: list = field(default_factory=list)
    mergers: list = field(default_factory=list)
    # extraction
    rfc: RidgeFilterConfig = None         # default: keep shallow lines, top 40
    max_ridge_pts: int = 30
    max_energy_span: float = 25.0         # eV; reject runaway linked ridges
    # model / cost
    hmax: int = HMAX_DEFAULT
    fix_disp: bool = True                 # after the rough fit, RE-estimate disp from the
                                          # ref-line slope-ratio (geometry-independent) and
                                          # CONFINE it to a +-disp_cone band around that in
                                          # the final refine (not hard-fixed - the ref-line
                                          # estimate has its own uncertainty, and a hard fix
                                          # let the cluster lines pull the rest off). Removes
                                          # the disp<->dY runaway while keeping a little play.
    disp_cone: float = 0.02               # fractional half-width of the disp band (+-2%)
    pin_dY: float = None                  # campaign input: hold dY at this (eye-identified)
                                          # value in the final refine. At fixed disp a flat
                                          # dY<->dR<->dRP valley remains that NO auto feature
                                          # breaks (ridges/anchors/merger all retune) - only
                                          # following the full steep lines by eye does. Set
                                          # this when you have read dY off the full lines.
    cap: float = 10.0                     # eV; per-ridge residual cap in the cost
    anchor_weight: float = 3.0            # anchored ridges: UNCAPPED x this weight.
                                          # Anchors are ground truth - capping them lets the
                                          # optimizer trade them against ordinary ridges and
                                          # settle in a wrong basin with a good-looking cost.
    tol: float = 6.0                      # eV; per-ridge pass tolerance for reporting
    plane_prune_window: float = 200.0     # eV; generous (Ecorr can be +-150)
    # grid stage (seedless): spans are ABOUT the prior dR/dY; tilt is absolute.
    # disp and Ecorr are ALSO scanned per node (cheap: the model is geometry-only) -
    # the stage-2 estimates are only coarse seeds, since the reference-line model slope
    # itself depends on the unknown geometry (a bare prior can bias disp by ~30%).
    grid_dR: np.ndarray = None            # default prior[1] + [-1.2 .. 1.2] step 0.6
    grid_dY: np.ndarray = None            # default prior[2] + [-1.2 .. 1.2] step 0.6
    grid_tilt: tuple = (-0.03, 0.0, 0.03)
    grid_disp_factors: tuple = (0.75, 0.875, 1.0, 1.125)   # x stage-2 disp estimate
    ecorr_halfwidth: float = 25.0         # eV about the stage-2 estimate, step 2

    def __post_init__(self):
        if self.rfc is None:
            self.rfc = RidgeFilterConfig(min_abs_slope_dE_dangle=None, top_n_ridges=40)
        if self.ref_scans is None:
            self.ref_scans = list(self.scan_files)
        if self.grid_dR is None:
            self.grid_dR = self.prior[1] + np.arange(-1.2, 1.21, 0.6)
        if self.grid_dY is None:
            self.grid_dY = self.prior[2] + np.arange(-1.2, 1.21, 0.6)


# ------------------------------------------------------------------- stage 1: extract --

def _reject_outliers(a, e, S=150.0, abs_tol=6.0, k=4.0, min_keep=4):
    """Drop lone points that sit off a ridge's own line (e.g. a single bright pixel from a
    CROSSING line that the linker stitched in). Robust total-least-squares: PCA the points in
    (S*pitch, energy) space (S scales pitch->eV so near-vertical and shallow lines are handled
    alike), reject points whose PERPENDICULAR distance to the principal axis exceeds
    max(abs_tol, k*MAD). Returns the kept (a, e) (unchanged if too few to judge)."""
    a = np.asarray(a, float); e = np.asarray(e, float)
    if len(a) < min_keep + 2:
        return a, e
    pts = np.column_stack([a * S, e]); c = pts.mean(0); d = pts - c
    try:
        _, _, vt = np.linalg.svd(d, full_matrices=False)
    except Exception:
        return a, e
    axis = vt[0]; nrm = np.array([-axis[1], axis[0]])     # perpendicular direction
    perp = np.abs(d @ nrm)
    mad = float(np.median(np.abs(perp - np.median(perp)))) * 1.4826
    keep = perp <= max(abs_tol, k * mad)
    return (a[keep], e[keep]) if keep.sum() >= min_keep else (a, e)


def extract(camp, verbose=True):
    """Ridges + candidate planes per scan. Anchor/merger planes are ALWAYS kept in the
    candidate set (pruning at a wrong prior once dropped the merger planes entirely)."""
    must_keep = [h_to_array(a.H) for a in camp.anchors]
    for m in camp.mergers:
        must_keep += [h_to_array(m.H1), h_to_array(m.H2)]
    sets = []
    for fn in camp.scan_files:
        loaded = camp.load(fn)
        img, phen, pitch, fixed = loaded[:4]
        axis = loaded[4] if len(loaded) > 4 else "pitch"
        pk = detect_peak_candidates(img, phen, pitch, scan_id=fn[:13])
        rp, rg = link_ridge_candidates(pk)
        frp, frg = filter_ridge_candidates(rp, rg, config=camp.rfc)
        keep = frg.loc[(frg["energy_max"] - frg["energy_min"]) <= camp.max_energy_span,
                       "ridge_id"]
        frp = frp[frp["ridge_id"].isin(keep)]
        ridges = []
        for _, g in frp.groupby("ridge_id"):
            g = g.sort_values("angle")
            a = g["angle"].to_numpy(float); e = g["energy"].to_numpy(float)
            a, e = _reject_outliers(a, e)         # drop crossing-line stragglers
            n_orig = len(a)                       # pre-subsample length (ref-ridge pick)
            if len(a) > camp.max_ridge_pts:
                i = np.linspace(0, len(a) - 1, camp.max_ridge_pts).round().astype(int)
                a, e = a[i], e[i]
            sl = float(np.polyfit(a, e, 1)[0]) if len(a) > 1 else 0.0
            pm = float(np.median(a))
            H = next((h_to_array(an.H) for an in camp.anchors if an.matches(fn, pm, sl)), None)
            mg = next((m for m in camp.mergers if m.matches(fn, pm, sl)), None)
            ridges.append(dict(a=a, e=e, w=float(g["prominence"].sum()), pm=pm, sl=sl,
                               n_orig=n_orig, H=H, merger=mg))
        # candidate planes at the prior, pruned GENEROUSLY; force-include anchored planes
        geom0 = (*camp.prior, 0.0, 0.0)
        _, _, Hs = _render(geom0, axis, fixed, np.array([phen[0] - 200, phen[-1] + 200]),
                           np.linspace(pitch.min(), pitch.max(), 12), hmax=camp.hmax)
        Harrs = [h_to_array(H) for H in Hs]
        if ridges:
            emin = min(r["e"].min() for r in ridges); emax = max(r["e"].max() for r in ridges)
            amid = np.array([np.mean([r["pm"] for r in ridges])])
            kept = []
            for Ha in Harrs:
                try:
                    em = float(_E_line(Ha, amid, fixed, axis, geom0)[0])
                except Exception:
                    continue
                if (emin - camp.plane_prune_window) <= em <= (emax + camp.plane_prune_window):
                    kept.append(Ha)
            Harrs = kept or Harrs
        for Ha in must_keep:
            if not any(np.array_equal(Ha, K) for K in Harrs):
                Harrs.append(Ha)
        sets.append(dict(fn=fn, fixed=fixed, axis=axis, ridges=ridges, Harrs=Harrs,
                         prange=(float(pitch.min()), float(pitch.max())),
                         img=img, phen=phen, ang=pitch))   # kept for full-line tracing
        if verbose:
            na = sum(r["H"] is not None for r in ridges)
            nm = sum(r["merger"] is not None for r in ridges)
            print(f"  {fn[:17]}: {len(ridges)} ridges ({na} anchored, {nm} merger), "
                  f"{len(Harrs)} planes, pitch {pitch.min():.1f}-{pitch.max():.1f}")
    return sets


def prune_anchors(camp, sets, p7, pivot, tol=6.0, verbose=True):
    """Drop ANCHORED ridges the (rough) geometry cannot place within `tol` perp eV. An anchor
    is a (pitch, slope) window; in crowded near-vertical regions it can grab a short fragment
    of a CROSSING line instead of the intended plane. Those internally-consistent wrong-line
    fragments survive per-ridge outlier rejection but fail their assigned plane -> drop them
    (in place). Returns the number dropped."""
    geom5, ec, disp = p7[:5], p7[5], p7[6]
    dropped = 0
    for s in sets:
        kept = []
        for r in s["ridges"]:
            if r["H"] is None:
                kept.append(r); continue
            e = pivot + disp * (r["e"] - pivot)
            em = _E_line(r["H"], r["a"], s["fixed"], s["axis"], geom5) + ec
            resid = float(_perp_resid(r["a"], e, em[None, :])[0])
            if resid <= tol:
                kept.append(r)
            else:
                dropped += 1
                if verbose:
                    print(f"    pruned mis-anchored {list(map(int, r['H']))} @ "
                          f"{s['fn'][9:17]} pitch {r['pm']:.2f} (resid {resid:.1f} eV)")
        s["ridges"] = kept
    return dropped


def update_planes(camp, sets, geom5, verbose=False):
    """Re-derive the candidate plane sets AT a given geometry (in place). Pruning at the
    bare prior is too aggressive once the fit has moved (a steep plane 200 eV away at
    the prior can be the RIGHT plane at the fit - e.g. [-1,-1,-1] at 36.6 deg). Called
    between pipeline stages as the geometry improves. Anchor/merger planes always kept."""
    must_keep = [h_to_array(a.H) for a in camp.anchors]
    for m in camp.mergers:
        must_keep += [h_to_array(m.H1), h_to_array(m.H2)]
    for s in sets:
        if not s["ridges"]:
            continue
        emin = min(r["e"].min() for r in s["ridges"])
        emax = max(r["e"].max() for r in s["ridges"])
        _, _, Hs = _render(geom5, s["axis"], s["fixed"],
                           np.array([emin - camp.plane_prune_window,
                                     emax + camp.plane_prune_window]),
                           np.linspace(*s["prange"], 12), hmax=camp.hmax)
        Harrs = [h_to_array(H) for H in Hs]
        for Ha in must_keep:
            if not any(np.array_equal(Ha, K) for K in Harrs):
                Harrs.append(Ha)
        if verbose:
            print(f"  {s['fn'][:17]}: {len(s['Harrs'])} -> {len(Harrs)} planes")
        s["Harrs"] = Harrs
    return sets


# ------------------------------------------------------- stage 2: dispersion estimate --

def _disp_ec_at(camp, sets, geom5, pivot=None):
    """disp (from the `ref_plane` slope-ratio) and Ecorr (from its offset) at a GIVEN geometry.
    The reference line is long & shallow, so its model slope is ~insensitive to the angular
    geometry but its slope-RATIO pins disp, and its offset pins Ecorr. Called per grid node so
    disp/Ecorr are re-locked analytically instead of grid-searched. Returns (disp, ec, pivot)
    (disp/ec None if the ref line isn't usable at this geometry)."""
    Href = h_to_array(camp.ref_plane)
    disps, ecs, piv = [], [], pivot
    for s in sets:
        if s["axis"] != "pitch" or not any(rs in s["fn"] for rs in camp.ref_scans):
            continue
        if not s["ridges"]:
            continue
        r = max(s["ridges"], key=lambda r: r["n_orig"])   # longest BEFORE subsampling
        sm = float(np.polyfit(r["a"], r["e"], 1)[0])
        if abs(sm) < 3:
            continue
        em = _E_line(Href, r["a"], s["fixed"], "pitch", geom5)
        if not np.all(np.isfinite(em)):
            continue
        d = float(np.polyfit(r["a"], em, 1)[0]) / sm
        if piv is None:
            piv = float(np.median(r["e"]))
        disps.append(d)
        ecs.append(float(np.median(piv + d * (r["e"] - piv) - em)))
    if not disps:
        return None, None, piv
    return float(np.median(disps)), float(np.median(ecs)), piv


def estimate_disp(camp, sets):
    """disp/pivot/Ecorr at the prior geometry (seed for the grid). See _disp_ec_at."""
    disp, ec, piv = _disp_ec_at(camp, sets, (*camp.prior, 0.0, 0.0))
    return disp, piv, ec


def fix_disp_from_ref(camp, sets, geom5):
    """disp = median(model_slope / measured_slope) of the reference line, evaluated AT a
    given (rough) geometry. The shallow reference line is dY/dR-insensitive, so this is
    geometry-independent to ~0.001 across the whole valley - a rock-solid disp anchor that
    removes the disp<->dY trade. Returns disp (None if no usable ref ridge)."""
    Href = h_to_array(camp.ref_plane)
    ds = []
    for s in sets:
        if s["axis"] != "pitch" or not any(rs in s["fn"] for rs in camp.ref_scans):
            continue
        cand = [r for r in s["ridges"]
                if r["n_orig"] >= 40 and abs(np.polyfit(r["a"], r["e"], 1)[0]) < 60]
        if not cand:
            continue
        r = max(cand, key=lambda r: r["n_orig"])
        sm = float(np.polyfit(r["a"], r["e"], 1)[0])
        if abs(sm) < 3:
            continue
        smod = float(np.polyfit(r["a"], _E_line(Href, r["a"], s["fixed"], "pitch", geom5), 1)[0])
        ds.append(smod / sm)
    return float(np.median(ds)) if ds else None


# -------------------------------------------------------------- fast model evaluation --

def _flatten(sets, thin=None):
    """Flatten to per-ridge tuples; optionally thin to `thin` points (grid stage)."""
    R, M = [], []
    for si, s in enumerate(sets):
        for r in s["ridges"]:
            a, e = r["a"], r["e"]
            if thin and len(a) > thin:
                i = np.linspace(0, len(a) - 1, thin).round().astype(int)
                a, e = a[i], e[i]
            t = (s["fixed"], s["axis"], si, a, e, r["H"], r["pm"])
            (M if r["merger"] is not None else R).append(
                t + ((r["merger"],) if r["merger"] is not None else ()))
    return R, M


def _emat(geom5, sets, R, M):
    """Model energies for every (ridge, candidate-plane) pair at one geometry. The model
    depends ONLY on geometry; Ecorr/disp are applied cheaply downstream, so an Ecorr grid
    or trace costs nothing extra. Returns (per-ridge Emat list, per-merger (E1, E2) list).
    """
    out_R, out_M = [], []
    c = _bm.cryst
    _bm.autoUpdate = False
    try:
        # delta_*_INITIAL convention (mount pre-rotation) - bit-identical to render_pitch /
        # bragg_energy. (delta_* would fold the offsets into the tilted measurement rotation
        # -> up to ~23 eV error for l=+-1 planes when dRP/dYP!=0; see bragg_energy docstring.)
        c.delta_pitch = 0.0; c.delta_roll = 0.0; c.delta_yaw = 0.0
        c.delta_pitch_initial = np.deg2rad(-geom5[0])
        c.delta_roll_initial = np.deg2rad(-geom5[1])
        c.delta_yaw_initial = np.deg2rad(-geom5[2])
        pa = np.array([1.0, -geom5[3], -geom5[4]])
        c.pitch_axes = pa / np.linalg.norm(pa)

        def _ept(v, fixed, axis, planes):
            if axis == "pitch":
                c.pitch, c.roll = np.deg2rad(v), np.deg2rad(fixed)
            else:
                c.pitch, c.roll = np.deg2rad(fixed), np.deg2rad(v)
            c.yaw = 0.0
            c.calc_RLab2Cryst()
            row = np.empty(len(planes))
            for hi, Ha in enumerate(planes):
                try:
                    row[hi] = _bm.bragg_wavelength(exact=False, H=Ha)[1]
                except Exception:
                    row[hi] = np.nan
            return row

        for fixed, axis, si, a, e, H, pm in R:
            planes = [H] if H is not None else sets[si]["Harrs"]
            out_R.append(np.column_stack([_ept(v, fixed, axis, planes) for v in a]))
        for fixed, axis, si, a, e, H, pm, mg in M:
            planes = [h_to_array(mg.H1), h_to_array(mg.H2)]
            out_M.append(np.column_stack([_ept(v, fixed, axis, planes) for v in a]))
    finally:
        _bm.autoUpdate = True
        c.delta_pitch_initial = 0.0; c.delta_roll_initial = 0.0; c.delta_yaw_initial = 0.0
    return out_R, out_M


REF_SLOPE = 150.0   # eV/deg: pitch<->energy scale of the perpendicular residual. Lines with
                    # |slope| << REF_SLOPE are scored by their ENERGY offset (as before); much
                    # steeper (near-vertical) lines by their PITCH offset x REF_SLOPE, so a
                    # 0.01 deg pitch mismatch no longer reads as ~15 eV. This is the point->line
                    # perpendicular distance in the (REF_SLOPE*pitch, energy) plane.


def _perp_resid(a, ed, em_mat, S=REF_SLOPE):
    """Per-plane median PERPENDICULAR residual (eV). em_mat: (nplanes, npts) model energies
    (incl Ecorr); ed, a: (npts) data energy and pitch. = |ed-em| / sqrt(1 + (slope/S)^2)."""
    a = np.asarray(a, float); n = len(a)
    if n >= 2:
        da = np.empty(n)
        da[1:-1] = a[2:] - a[:-2]; da[0] = a[1] - a[0]; da[-1] = a[-1] - a[-2]
        da = np.where(np.abs(da) < 1e-6, 1e-6, da)
        de = np.empty_like(em_mat, dtype=float)
        de[:, 1:-1] = em_mat[:, 2:] - em_mat[:, :-2]
        de[:, 0] = em_mat[:, 1] - em_mat[:, 0]; de[:, -1] = em_mat[:, -1] - em_mat[:, -2]
        m = de / da                                   # local model slope per point, per plane
    else:
        m = np.zeros_like(em_mat, dtype=float)
    perp = np.abs(ed[None, :] - em_mat) / np.sqrt(1.0 + (m / S) ** 2)
    return np.nanmedian(perp, axis=1)


def _resid(R, M, ematR, ematM, ec, disp, pivot, cap, anchor_weight=3.0):
    """(per-ridge residuals, per-merger residuals, cost). Residual is the steep-aware
    PERPENDICULAR distance (see _perp_resid) so near-vertical lines aren't penalised for tiny
    pitch mismatches. Ordinary ridges: equal weight, capped at `cap`. ANCHORED ridges (ground
    truth): UNCAPPED x anchor_weight. Mergers: uncapped x mg.weight."""
    res = np.empty(len(R))
    num = den = 0.0
    for k, (fixed, axis, si, a, e, H, pm) in enumerate(R):
        ed = pivot + disp * (e - pivot)
        med = _perp_resid(a, ed, ematR[k] + ec)
        res[k] = np.nanmin(med) if np.isfinite(med).any() else cap * 10
        if H is not None:
            num += anchor_weight * res[k]; den += anchor_weight
        else:
            num += min(res[k], cap); den += 1.0
    mres = []
    for k, (fixed, axis, si, a, e, H, pm, mg) in enumerate(M):
        ed = pivot + disp * (e - pivot)
        on_data = float(np.nanmax(_perp_resid(a, ed, ematM[k] + ec)))   # BOTH planes on data
        # model SPLITTING |E1-E2| at the merger ridge: a pure crossing-position measure
        # (immune to Ecorr/disp). The on-data term alone dilutes a displaced crossing -
        # a rival geometry put the crossing 0.6 deg off while keeping on_data moderate.
        split = float(np.nanmedian(np.abs(ematM[k][0] - ematM[k][1])))
        mres.append(on_data + split)
        num += mg.weight * mres[-1]; den += mg.weight
    return res, mres, float(num / den)


# ----------------------------------------------------------- full-line trace residual --
# Ridge FRAGMENTS leave a near-degenerate valley (dY <-> dRP <-> disp all trade off at the
# ~1 eV fragment level). The expert method - "follow the FULL visible line, not only the
# extracted ridges" - discriminates: the long shallow lines ([-1,-1,5], [-4,-2,2]) traced
# over their full extent separate valley members at >10 sigma. These functions encode it.

# ----------------------------------------------------------------- stage 3: grid scan --

def grid_search(camp, sets, disp, pivot, ec0, verbose=True):
    """Seedless coarse scan over (dR, dY, dRP, dYP) at dP = prior. Per node, disp and Ecorr
    are RE-LOCKED analytically from the reference-line slope-ratio/offset (one value each,
    geometry-correct) instead of grid-searched - far faster than the old 4xN disp/Ecorr loop
    and more accurate. Returns ranked seeds."""
    R, M = _flatten(sets, thin=8)
    # score nodes on the ANCHORED ridges only (1 plane each -> cheap; they carry the
    # ground-truth constraints) + the merger; disp/Ecorr are locked to the reference line.
    # The dense non-anchored ridges (many candidate planes each) are reintroduced in refine.
    Ra = [r for r in R if r[5] is not None]
    R = Ra if Ra else R
    nodes = list(itertools.product(camp.grid_dR, camp.grid_dY,
                                   camp.grid_tilt, camp.grid_tilt))
    if verbose:
        print(f"  grid: {len(nodes)} nodes, disp/Ecorr relocked per node from "
              f"{list(camp.ref_plane)} ({len(R)} anchored ridges, {len(M)} merger)")
    out, best = [], None
    for dR, dY, dRP, dYP in nodes:
        g = (camp.prior[0], dR, dY, dRP, dYP)
        try:
            eR, eM = _emat(g, sets, R, M)
            dv, ec, _ = _disp_ec_at(camp, sets, g, pivot)
        except Exception:
            continue
        if dv is None or not (0.4 < dv < 1.4):     # ref line unusable / nonphysical disp here
            continue
        _, _, c = _resid(R, M, eR, eM, ec, dv, pivot, camp.cap, camp.anchor_weight)
        out.append((c, [*g, ec, dv]))
        if verbose and (best is None or c < best):
            best = c
            print(f"    * cost={c:6.3f}  dR={dR:+.2f} dY={dY:+.2f} "
                  f"dRP={dRP:+.3f} dYP={dYP:+.3f} Ec={ec:+.1f} disp={dv:.3f}")
    out.sort(key=lambda t: t[0])
    return out


# -------------------------------------------------------------------- stage 4: refine --

REFINE_STEPS = np.array([0.01, 0.05, 0.05, 0.003, 0.003, 1.5, 0.01])   # explicit simplex!


def refine(camp, sets, p7, pivot, maxiter=900, verbose=True, fixed=None, disp_bounds=None):
    """Nelder-Mead on (dP,dR,dY,dRP,dYP,Ecorr,disp) with an EXPLICIT initial simplex
    (default simplex freezes 0.0-valued params - see module docstring). `fixed` =
    {index: value} holds parameters constant (e.g. {2: dY} to pin the eye-identified yaw).
    `disp_bounds` = (lo, hi) confines disp to a band (the +-disp_cone band around the
    ref-line estimate) instead of hard-fixing it."""
    from scipy.optimize import minimize
    R, M = _flatten(sets, thin=12)
    fixed = dict(fixed or {})
    p7 = list(p7)
    for i, v in fixed.items():
        p7[i] = v
    free = [i for i in range(7) if i not in fixed]
    dlo, dhi = disp_bounds if disp_bounds else (0.5, 1.5)

    def mk(pf):
        p = list(p7)
        for j, i in enumerate(free):
            p[i] = pf[j]
        return p

    def cost(pf):
        p = mk(pf)
        if not (dlo <= p[6] <= dhi) or abs(p[5] - p7[5]) > 25:
            return 1e4
        try:
            eR, eM = _emat(p[:5], sets, R, M)
        except Exception:
            return 1e4
        return _resid(R, M, eR, eM, p[5], p[6], pivot, camp.cap, camp.anchor_weight)[2]

    p0 = [p7[i] for i in free]
    steps = [REFINE_STEPS[i] for i in free]
    n = len(free)
    simplex = np.vstack([p0] + [[p0[j] + (steps[j] if j == k else 0.0) for j in range(n)]
                                for k in range(n)])
    r = minimize(cost, p0, method="Nelder-Mead",
                 options=dict(initial_simplex=simplex, xatol=3e-4, fatol=3e-4,
                              maxiter=maxiter))
    x = mk(list(r.x))
    if verbose:
        tag = "".join(f" [{'dP dR dY dRP dYP Ec disp'.split()[i]} fixed]" for i in fixed)
        print(f"  refine: cost={r.fun:.3f} ({r.nit} iters)  dP={x[0]:.3f} dR={x[1]:.3f} "
              f"dY={x[2]:.3f} dRP={x[3]:+.4f} dYP={x[4]:+.4f} Ec={x[5]:+.2f} "
              f"disp={x[6]:.4f}{tag}")
    return x, float(r.fun)


# ------------------------------------------------------------- stage 5: centroid Ecorr --

def centroid_ecorr(camp, sets, p7, pivot, max_model_slope=60.0, resid_gate=4.0,
                   ewin=12.0, verbose=True):
    """Ecorr polish with the geometry FIXED. The detected peak points can sit ~0.5-1 eV off
    the visible line centers, so the final Ecorr is anchored on the brightness CENTROID of
    the image along each well-fit SHALLOW line (|model slope| < max_model_slope: steep
    lines mix pitch error into the energy residual). Returns the polished p7."""
    geom, ec, disp = p7[:5], p7[5], p7[6]
    R, M = _flatten(sets)
    eR, eM = _emat(geom, sets, R, M)
    biases = []
    img_cache = {}
    for k, (fixed, axis, si, a, e, H, pm) in enumerate(R):
        ed = pivot + disp * (e - pivot)
        med = _perp_resid(a, ed, eR[k] + ec)
        if not np.isfinite(med).any() or np.nanmin(med) > resid_gate:
            continue
        planes = [H] if H is not None else sets[si]["Harrs"]
        Ha = planes[int(np.nanargmin(med))]
        fn = sets[si]["fn"]
        if fn not in img_cache:
            img_cache[fn] = camp.load(fn)[:3]
        img, phen, pitch = img_cache[fn]
        phen_c = pivot + disp * (phen - pivot)
        m = (pitch >= a.min()) & (pitch <= a.max())
        P, C = pitch[m], img[m]
        em = _E_line(Ha, P, fixed, axis, geom) + ec
        if np.nanmedian(np.abs(np.gradient(em, P))) > max_model_slope:
            continue
        for i in range(len(P)):
            if not np.isfinite(em[i]):
                continue
            w = (phen_c >= em[i] - ewin) & (phen_c <= em[i] + ewin)
            if w.sum() < 3:
                continue
            prof = C[i][w]; ee = phen_c[w]; j = int(np.argmax(prof))
            k0, k1 = max(0, j - 3), min(len(ee), j + 4)
            biases.append(float(np.sum(ee[k0:k1] * prof[k0:k1]) / np.sum(prof[k0:k1])
                                - em[i]))
    shift = float(np.median(biases)) if biases else 0.0
    if verbose:
        print(f"  centroid polish: {len(biases)} columns, bias={shift:+.2f} eV "
              f"-> Ecorr {ec:+.2f} -> {ec + shift:+.2f}")
    return [*geom, ec + shift, disp]


# ------------------------------------------------------------------ stage 6: validate --

def validate(camp, sets, p7, pivot, verbose=True):
    """Per-ridge report + merger crossing check. Returns a result dict (also saved by
    run())."""
    geom, ec, disp = p7[:5], p7[5], p7[6]
    R, M = _flatten(sets)
    eR, eM = _emat(geom, sets, R, M)
    res, mres, cost = _resid(R, M, eR, eM, ec, disp, pivot, camp.cap, camp.anchor_weight)
    passf = float((res < camp.tol).mean())
    rows = []
    for k, (fixed, axis, si, a, e, H, pm) in enumerate(R):
        ed = pivot + disp * (e - pivot)
        med = _perp_resid(a, ed, eR[k] + ec)
        planes = [H] if H is not None else sets[si]["Harrs"]
        Ha = planes[int(np.nanargmin(med))] if np.isfinite(med).any() else None
        rows.append(dict(scan=sets[si]["fn"][:17], axis=axis, pitch=pm,
                         plane=None if Ha is None else [int(v) for v in Ha],
                         anchored=H is not None, resid=float(res[k])))
    mergers = []
    for k, (fixed, axis, si, a, e, H, pm, mg) in enumerate(M):
        span = 12.0 if axis == "pitch" else 0.2
        P = np.linspace(pm - span, pm + span, 241)
        d = (_E_line(mg.H1, P, fixed, axis, geom)
             - _E_line(mg.H2, P, fixed, axis, geom))
        cr = np.where(np.diff(np.sign(d)) != 0)[0]
        mp = float(P[cr[0]] - d[cr[0]] * (P[cr[0] + 1] - P[cr[0]])
                   / (d[cr[0] + 1] - d[cr[0]])) if len(cr) else None
        mergers.append(dict(pitch_data=pm, pitch_model=mp, resid=mres[k]))
    if verbose:
        npass = int((res < camp.tol).sum())
        print(f"  validate: pass(<{camp.tol:.0f} eV) {npass}/{len(res)} = {passf:.2f}, "
              f"median={np.median(res):.2f} eV, cost={cost:.3f}")
        for mg in mergers:
            print(f"  merger: data {mg['pitch_data']:.2f} vs model "
                  f"{mg['pitch_model'] if mg['pitch_model'] else float('nan'):.2f} deg, "
                  f"resid={mg['resid']:.1f} eV")
        for r in sorted(rows, key=lambda r: r["pitch"]):
            if r["resid"] >= camp.tol:
                print(f"    FAIL pitch {r['pitch']:5.1f} {r['plane']} resid={r['resid']:.1f}")
    return dict(params=dict(dP=p7[0], dR=p7[1], dY=p7[2], dRP=p7[3], dYP=p7[4],
                            Ecorr=p7[5], disp=p7[6], pivot=pivot),
                cost=cost, passf=passf, median=float(np.median(res)),
                ridges=rows, mergers=mergers)


def plot_overlay(camp, p7, pivot, scan_file, out, title=None):
    """Labeled model-vs-data overlay in the dispersion-corrected frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    geom, ec, disp = p7[:5], p7[5], p7[6]
    loaded = camp.load(scan_file)
    img, phen, pitch, fixed = loaded[:4]
    axis = loaded[4] if len(loaded) > 4 else "pitch"
    phen_c = pivot + disp * (phen - pivot)
    ag = np.linspace(pitch.min(), pitch.max(), 300)
    # plane LIST from render_pitch; ENERGIES from _E_line (same delta_*_initial convention,
    # so identical to render_pitch but without its angle-range-dependent curve labelling).
    # This guarantees the plotted lines are exactly what the fit evaluates. pit == ag here.
    _, _, Hs = _render(geom, axis, fixed, phen, ag, hmax=camp.hmax)
    pit = ag
    phenH = np.column_stack([_E_line(h_to_array(H), ag, fixed, axis, geom) for H in Hs]) \
        if Hs else np.empty((len(ag), 0))
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.pcolormesh(pitch, phen_c, img.T, shading="auto", vmax=np.percentile(img, 99.5))
    # extracted ridge points (cyan) - the visual cue for where the data lines actually are
    try:
        pk = detect_peak_candidates(img, phen, pitch, scan_id="ov")
        rp, rg = link_ridge_candidates(pk)
        frp, frg = filter_ridge_candidates(rp, rg, config=camp.rfc)
        ax.scatter(frp["angle"], pivot + disp * (frp["energy"].to_numpy() - pivot),
                   s=8, c="cyan", zorder=3, label="ridges")
    except Exception:
        pass
    cmap = plt.cm.tab10; n = 0
    for j in range(phenH.shape[1]):
        c = phenH[:, j] + ec
        inf = np.isfinite(c) & (c >= phen_c.min()) & (c <= phen_c.max())
        if inf.sum() < 2:
            continue
        col = cmap(n % 10); n += 1
        ax.plot(pit, c, lw=1.2, color=col, zorder=2)
        k = np.where(inf)[0][-1]
        lab = "[" + ",".join(str(int(v)) for v in np.asarray(Hs[j])) + "]"
        ax.annotate(lab, (pit[k], c[k]), fontsize=8, color=col, weight="bold",
                    xytext=(3, 0), textcoords="offset points", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
    ax.set_xlim(pitch.min(), pitch.max()); ax.set_ylim(phen_c.min(), phen_c.max())
    ax.set_xlabel(f"{axis} [deg]"); ax.set_ylabel("E (disp-corrected) [eV]")
    ax.set_title(title or f"{camp.name} {scan_file[:17]}")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out} ({n} lines)")


# --------------------------------------------------------------------- orchestration --

def run(camp, n_refine_seeds=3, out_json=None, overlay_scans=None, out_dir="", verbose=True):
    """Full pipeline. Returns the validate() result dict (params under 'params').
    `out_dir` is prepended to out_json and the overlay filenames (the campaign folder)."""
    print(f"[1/6] extract ridges ({camp.name})")
    sets = extract(camp, verbose=verbose)
    print(f"[2/6] dispersion + Ecorr estimate from {list(camp.ref_plane)}")
    disp, pivot, ec0 = estimate_disp(camp, sets)
    print(f"  disp={disp:.3f}  pivot={pivot:.1f}  Ecorr~{ec0:+.1f}")
    print(f"[3/6] seedless grid (dR, dY, dRP, dYP)")
    seeds = grid_search(camp, sets, disp, pivot, ec0, verbose=verbose)
    print(f"[4/6] refine top {n_refine_seeds} seeds (7 params, explicit simplex)")
    # re-derive candidate planes at the best grid geometry (prior pruning is stale now)
    update_planes(camp, sets, seeds[0][1][:5])
    best = None
    for c0, p7 in seeds[:n_refine_seeds]:
        x, c = refine(camp, sets, p7, pivot, verbose=verbose)
        if best is None or c < best[1]:
            best = (x, c)
    x = best[0]
    update_planes(camp, sets, x[:5])
    # prune anchored ridges the rough geometry can't place (crossing-line fragments the anchor
    # windows grabbed in crowded near-vertical regions), then re-refine on the cleaned set.
    nd = prune_anchors(camp, sets, x, pivot, tol=camp.tol, verbose=verbose)
    if nd:
        print(f"  pruned {nd} mis-anchored ridge(s); re-refining")
        x, c = refine(camp, sets, x, pivot, verbose=verbose)
        update_planes(camp, sets, x[:5])
    # disp-cone: the reference-line slope-ratio pins disp geometry-independently; confine the
    # final refine to a +-disp_cone band around it (not hard-fixed) so the disp<->dY runaway
    # is removed but disp keeps a little physical play.
    fixed = {}
    disp_bounds = None
    if camp.fix_disp:
        d = fix_disp_from_ref(camp, sets, x[:5])
        if d is not None:
            disp_bounds = (d * (1 - camp.disp_cone), d * (1 + camp.disp_cone))
            x[6] = min(max(x[6], disp_bounds[0]), disp_bounds[1])
            print(f"  disp confined to {list(camp.ref_plane)} slope-ratio {d:.4f} "
                  f"+-{camp.disp_cone*100:.0f}%: [{disp_bounds[0]:.4f}, {disp_bounds[1]:.4f}]")
    # pin_dY: a flat dY<->dR<->dRP valley remains (ridges, anchors AND the merger position all
    # retune across dY) - NOT machine-breakable from extracted features. Hold dY only if it
    # was read off the FULL steep lines by eye.
    if camp.pin_dY is not None:
        fixed[2] = camp.pin_dY; x[2] = camp.pin_dY
        print(f"  dY pinned (eye-identified): dY={camp.pin_dY:+.3f}")
    if fixed or disp_bounds:
        x, c = refine(camp, sets, x, pivot, verbose=verbose, fixed=fixed, disp_bounds=disp_bounds)
        update_planes(camp, sets, x[:5])
        x, c = refine(camp, sets, x, pivot, verbose=verbose, fixed=fixed, disp_bounds=disp_bounds)
    print(f"[5/6] centroid Ecorr polish (geometry fixed)")
    x = centroid_ecorr(camp, sets, x, pivot, verbose=verbose)
    print(f"[6/6] validate (planes re-derived at the final geometry)")
    update_planes(camp, sets, x[:5])
    result = validate(camp, sets, x, pivot, verbose=verbose)
    if out_json:
        path = os.path.join(out_dir, out_json) if out_dir else out_json
        with open(path, "w") as f:
            json.dump(result, f, indent=1)
        print(f"  saved {path}")
    for fn in (overlay_scans or []):
        out = f"overlay_{fn[:17]}.png"
        plot_overlay(camp, x, pivot, fn,
                     out=os.path.join(out_dir, out) if out_dir else out,
                     title=f"{camp.name} {fn[:17]}  pass={result['passf']:.2f}")
    return result
