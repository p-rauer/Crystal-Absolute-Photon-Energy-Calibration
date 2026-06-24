# Crystal Calibration Recipe (consolidated, June 2026)

Reusable pipeline: `functions/calibration_pipeline.py`.
Campaign example: `calibrate_apr2026.py` (apr2026, 9 keV).
Validated on oct2024 (12.4 keV) and apr2026 (9 keV, chamber moved between campaigns).

## What you provide per campaign (the `Campaign` dataclass)

1. **A loader** `load(fn) -> (image[n_pitch, n_energy], energy_raw, pitch_deg, roll_deg)`.
   All format quirks live here (motor -> angle conversions, time-dependent energy
   zero-points, roll from status files, energy-region crop, sample sorting).
2. **Scan files** spanning a WIDE pitch range. The axis tilt (dRP, dYP) couples pitch
   into effective roll/yaw offsets; without low+high pitch groups it is unconstrained
   and the fit will silently absorb the error into dR/dY.
3. **A rough prior** `(dP, dR, dY)`. Only dP needs to be approximately right (it comes
   from the known motor offset). dR = dY = 0 is fine - the grid stage finds the basin.
4. **A reference plane** (e.g. `[-1,-1,5]`): the long shallow line used to pre-estimate
   dispersion and Ecorr. Shallow lines are robust to the angular mis-calibration.
5. **A few hand-picked `PlaneAnchor`s** - the only real manual physics input, and the
   single most decisive ingredient (see "Why anchors" below).
6. **Optionally `LineMerger`s** - points where two planes visibly merge into one ridge.

## Pipeline stages

| # | Stage | What it does | Key conventions |
|---|-------|--------------|-----------------|
| 1 | extract | peaks (brightness gate, global sigma) -> ridge linking -> filtering | `min_abs_slope=None` - NEVER drop shallow lines |
| 2 | estimate | disp = model/measured slope of the reference line; pivot; Ecorr | only a coarse seed (model slope depends on the unknown geometry) |
| 3 | grid | coarse scan (dR, dY, dRP, dYP) x (disp, Ecorr) at fixed dP | disp/Ecorr scans are free: model is geometry-only |
| 4 | refine | Nelder-Mead, all 7 params, from the top grid seeds | EXPLICIT initial simplex (see pitfalls) |
| 4b | disp-fix + pin | re-fix disp from the reference-line slope ratio; optionally pin dY | `fix_disp`, `pin_dY` on the Campaign |
| 5 | centroid | Ecorr polish from image brightness centroids, geometry FIXED | shallow lines only (steep mixes pitch error into energy) |
| 6 | validate | per-ridge table, pass fraction, merger pitch check, overlays | tolerance 6 eV (apr2026 noise floor) |

## The model-evaluation convention (the bug that cost days)

The crystal model offers two ways to apply the mount offsets dP/dR/dY:
`delta_*_initial` (a fixed pre-rotation about UN-tilted axes — the "initial rotation in
mount" block of `calc_RLab2Cryst`) and `delta_*` (folded into the measurement rotation,
which then runs about the TILTED `pitch_axes`). They agree only at pitch 90 deg and
diverge linearly away from it, up to ~23 eV for yaw-sensitive (l=+-1) planes once
dRP/dYP!=0. Physically dP/dR/dY ARE mount misalignments, so `delta_*_initial` is correct;
it is also what `render_pitch` (and the plots) use. The whole pipeline — `bragg_energy`,
`_E_line`, `_emat`, `render_pitch`, `plot_overlay` — is pinned to `delta_*_initial` so the
fit and the plots evaluate identical physics. (For a long time the fit used `delta_*`
while the plots used `delta_*_initial`, so every fit silently fought the plots.)

## The degeneracy valley

The axis-tilt parameters act as effectively LINEAR terms in pitch:
`dY_eff(P) = dY + dYP*(P - 90 deg)` (and analogously dR/dRP). With pitch groups at only
two pitches the data pins two effective values, so the individual (dY, dYP) slide along a
line and dRP/disp join the trade-off. Ridge fragments fit ~equally well anywhere along
this valley. What pins it: (a) `fix_disp` removes the disp direction; (b) the merger
position is the sharpest remaining constraint; (c) failing that, `pin_dY` to the value
read off the FULL visible steep lines by eye. A wide roll scan (>+-0.3 deg) or a third
mid-pitch group would break it from data alone. The merger cost needs BOTH the distance of
the merged ridge to the two planes AND the model splitting |E1-E2| at the ridge (a
displaced crossing keeps the first moderate while violating the second).

## The objective ("October convention")

- **Every ridge counts equally** (cost = mean over ridges of the capped median
  residual). Not brightness-weighted; not per-scan: a dense high-pitch cluster must not
  outvote sparse low-pitch scans.
- Per ridge: best candidate plane unless anchored; residual capped (10 eV) so outliers
  do not dominate; **mergers uncapped + soft weight** (~2 ridges).
- Validation metric: fraction of ridges with median residual < 6 eV, plus visual
  overlays. A good global cost with bad overlays means mis-assignment, not success.

## Why anchors (the apr2026 lesson)

Near-vertical lines (|dE/dpitch| > 150 eV/deg) have enormous energy leverage: at a
slightly wrong geometry they sit **~200 eV away from their true plane**. Auto
best-plane matching then silently picks some other plane that happens to pass nearby
and reports a SMALL residual - the fit looks fine while dR/dY/tilt are badly wrong
(apr2026: dR appeared as 1.60 instead of 0.58, compensated by dY/dYP). A handful of
hand-identified anchors for exactly these steep lines makes the cost smooth and pulls
the geometry into the true basin. Conversely, anchored steep lines are the best
constraint on the angular parameters - identify them first when staring at a new
campaign:
- apr2026: `[-1,1,1]` (~35 deg), `[-1,1,-1]` (~38), `[-4,-2,2]` (~38.6, shallow),
  `[-2,0,-2]` (~75.2, outermost-left of the 75-76 cluster).

**Mergers**: where two planes (h<->k mirror pairs, e.g. `[-3,-1,1]`/`[-1,-3,1]`) merge
into ONE ridge (apr2026: pitch 66.1), pull BOTH planes onto that ridge. Enforce
*softly* (finite spectrometer/ridge resolution) and remember the crossing depends on
roll AND yaw - it is not a pure roll = 0 probe. This constraint exposed the wrong
dR/dRP that the line fit alone could not see.

## Pitfalls (each cost us real debugging time)

1. **scipy Nelder-Mead default simplex freezes 0.0-valued parameters** (step 2.5e-4).
   Ecorr started at 0.0 never moved -> a constant -1 eV bias. Always pass
   `initial_simplex` with physical steps (`REFINE_STEPS`).
2. **Detected peak points sit ~0.5-1 eV off the visible line centers.** The final Ecorr
   must come from image brightness centroids (stage 5), not from the detected points.
3. **Do not free all 7 params to "polish"**: the capped cost lets disp/dP drift to game
   the many high-pitch ridges while breaking the few steep anchors. Stage 5 therefore
   fixes the geometry and only shifts Ecorr.
4. **Plane pruning at a wrong prior can drop the right planes** (the merger pair was
   200 eV away and got pruned). Anchor/merger planes are force-kept; pruning window is
   generous (+-200 eV).
5. **Energy zero-points can be time-dependent** within one campaign (apr2026: 9080
   before 11:17, 9065 after - a 15 eV step). Check timestamps, put it in the loader.
6. **Dispersion correction is affine about a pivot**: `E = pivot + disp*(raw - pivot)`,
   apr2026: pivot 9046.82, disp 0.869 (true dE/px = 0.869 x 0.18 = 0.156). When moving
   the result elsewhere, transfer the pivot too: keeping a pixel-anchor (e.g.
   "pixel 1550 = 9065") while scaling dE/px shifts everything by ~2.4 eV.
7. **The model convention**: measured E = E_model + Ecorr; `pitch_axes=[1,-dRP,-dYP]`;
   mount offsets go in **delta_*_initial** (NOT delta_*) so the fit matches render_pitch
   exactly — verified 0.000 eV across planes (see "model-evaluation convention" above).
8. The C diffraction model is **not fork-safe** - parallelize with independent OS
   processes, never `multiprocessing`. Pin BLAS threads (`OMP_NUM_THREADS=1`...).

## Reference results

| campaign | dP | dR | dY | dRP | dYP | Ecorr | disp (pivot) | pass |
|----------|-----|-----|-----|------|------|-------|--------------|------|
| oct2024 (manual ref) | 1.153 | 2.129 | 0.755 | -0.016 | -0.003 | -6.5 | 0.700 (12407) | 0.94 |
| apr2026 | 1.342 | 0.576 | -0.099 | +0.0277 | -0.0001 | -1.01 | 0.869 (9046.8) | 0.93 |
