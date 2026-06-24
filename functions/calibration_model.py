"""Single source of truth for the crystal Bragg model + energy-axis conventions.

Every calibration experiment (template coarse stage, assignment fine fit,
diagnostics) must import render/axis helpers from HERE so they share identical
physics.  Re-implementing the model per script is what produced the convention
drift (a spurious +5 eV vs the physical −6.5 eV Ecorr) in earlier probes.

CONVENTIONS (verified against the manual fit in calibration_October_2024.py):

  measured energy axis:   E_meas[i] = (phen_scale[i] − Ecentral)·disp + Ecentral
                          with disp = dEdP/disp_orig = 0.105/0.15 = 0.7.
                          (phen_scale is linear at 0.1501 eV/px, so this equals
                           the notebook's index axis 0.105·(i−pcentral)+Ecentral
                           to <0.1 eV.)

  Bragg model output:     E_model(H, angle; geometry)  — the TRUE photon energy.

  energy-offset relation: E_meas = E_model + Ecorr     (Ecorr ≈ −6.47 eV)
                          i.e. residual = E_meas − (E_model + Ecorr),
                          and the measured ridge sits Ecorr below the model.

  geometry vector:        (dP, dR, dY, dRP, dYP)  in degrees; pitch_axes tilt
                          uses [1, −dRP, −dYP] (normalized).  delta_*_initial
                          and delta_* both take −value (the model REPLACES
                          delta_* with delta_*_initial after initialize(); they
                          are not additive — verified).
"""
from __future__ import annotations
import numpy as np
from dynamic_diffraction_module import crystal, bmirror

ECENTRAL = 12407.0
DEDP_CORRECTED = 0.105
DISP_ORIG = 0.15
DISP_DEFAULT = DEDP_CORRECTED / DISP_ORIG          # 0.7
HMAX_DEFAULT = 8                                    # match the manual fit's hkl range

# module-level model singleton
_cr = crystal(); _cr.set_crystal_planes("001")
_bm = bmirror(H0=[0, 0, 4], cryst=_cr)


def measured_energy_axis(phen_scale, disp=DISP_DEFAULT, ecentral=ECENTRAL):
    """Measured spectrometer energy axis at dispersion-scale `disp`."""
    return (np.asarray(phen_scale, float) - ecentral) * disp + ecentral


def _apply_geometry(dP, dR, dY, dRP, dYP):
    _bm.cryst.delta_pitch_initial = np.deg2rad(-dP)
    _bm.cryst.delta_roll_initial  = np.deg2rad(-dR)
    _bm.cryst.delta_yaw_initial   = np.deg2rad(-dY)
    pa = np.array([1.0, -dRP, -dYP]); _bm.cryst.pitch_axes = pa / np.linalg.norm(pa)


def h_to_array(H):
    """Normalize any H representation to an int numpy array."""
    if isinstance(H, np.ndarray):
        return H.astype(int)
    if isinstance(H, (list, tuple)):
        return np.array(H, dtype=int)
    s = str(H).strip().strip("[]")
    return np.array([int(x) for x in s.replace(",", " ").split() if x], dtype=int)


def render_pitch(geom, roll, ea, angle_grid, hmax=HMAX_DEFAULT):
    """All Bragg H-curves for a PITCH scan (roll fixed, pitch = angle_grid).

    Returns (pitch_deg [n], phenH [n, nH] true energy, Hs [nH]).
    """
    dP, dR, dY, dRP, dYP = geom
    _bm.autoUpdate = False
    try:
        _bm.cryst.roll = np.deg2rad(roll)
        _apply_geometry(dP, dR, dY, dRP, dYP)
        _bm.initialize()
    finally:
        _bm.autoUpdate = True
    phenH, Hs, pit, _, _ = _bm.BraggCurvesEvsAngles_inRange(
        [ea[0] - 200, ea[-1] + 200], np.deg2rad(angle_grid),
        hmax=hmax, kmax=hmax, lmax=hmax, numSampling=120)
    return np.rad2deg(pit), np.asarray(phenH), list(Hs)


def render_roll(geom, pitch, ea, angle_grid, hmax=HMAX_DEFAULT):
    """All Bragg H-curves for a ROLL scan (pitch fixed, roll = angle_grid).

    Returns (roll_deg [n], phenH [n, nH] true energy, Hs [nH]).
    """
    dP, dR, dY, dRP, dYP = geom
    _bm.autoUpdate = False
    try:
        _bm.cryst.pitch = np.deg2rad(pitch)
        _apply_geometry(dP, dR, dY, dRP, dYP)
        _bm.initialize()
    finally:
        _bm.autoUpdate = True
    phenH, Hs, _, rollH, _ = _bm.BraggCurvesEvsAngles_inRange(
        [ea[0] - 200, ea[-1] + 200], PitchRange=None, RollRange=np.deg2rad(angle_grid),
        hmax=hmax, kmax=hmax, lmax=hmax, numSampling=120)
    return np.rad2deg(rollH), np.asarray(phenH), list(Hs)


def bragg_energy(H, angles, roll, geom):
    """E_model(H, angle) for given geometry — TRUE energy, no Ecorr.

    Per-point evaluation (use for fitting once H is assigned). Bit-identical to render_pitch.

    CONVENTION (must match render_pitch): dP/dR/dY are crystal MOUNT misalignments, applied
    as a fixed pre-rotation about un-tilted axes via delta_*_INITIAL (calc_RLab2Cryst's
    "initial rotation in mount" block); the axis tilt dRP/dYP tilts the pitch-rotation axis
    via pitch_axes. The alternative (delta_*, non-initial) folds the mount offsets into the
    TILTED measurement rotation and disagrees with render_pitch by up to ~23 eV for
    yaw-sensitive (l=+-1) planes once dRP/dYP!=0 — it is physically wrong. Keep this and
    render_pitch on delta_*_initial so the fit and the plots evaluate identical physics.
    """
    dP, dR, dY, dRP, dYP = geom
    H = h_to_array(H)
    angles = np.atleast_1d(np.asarray(angles, float))
    out = np.empty(angles.size)
    c = _bm.cryst
    _bm.autoUpdate = False
    try:
        c.delta_pitch = 0.0; c.delta_roll = 0.0; c.delta_yaw = 0.0
        c.delta_pitch_initial = np.deg2rad(-dP)
        c.delta_roll_initial  = np.deg2rad(-dR)
        c.delta_yaw_initial   = np.deg2rad(-dY)
        pa = np.array([1.0, -dRP, -dYP]); c.pitch_axes = pa / np.linalg.norm(pa)
        for i, p in enumerate(angles):
            c.pitch, c.roll, c.yaw = np.deg2rad(p), np.deg2rad(roll), 0.0
            c.calc_RLab2Cryst()
            out[i] = _bm.bragg_wavelength(exact=False, H=H)[1]
    finally:
        _bm.autoUpdate = True
        c.delta_pitch_initial = 0.0; c.delta_roll_initial = 0.0; c.delta_yaw_initial = 0.0
    return out


# Manual best-fit reference (calibration_October_2024.py), for verification only.
MANUAL_GEOM = (1.15333221, 2.12894512, 0.75484974, -0.01594835, -0.00269419)
MANUAL_ECORR = -6.46557482
