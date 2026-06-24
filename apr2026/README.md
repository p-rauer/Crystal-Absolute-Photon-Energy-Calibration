# apr2026 calibration (9 keV, SA2)

Worked example of the reusable calibration pipeline (`functions/calibration_pipeline.py`,
documented in `../Calibration_Recipe.md`). The chamber was moved vs oct2024, so this is a
fresh geometry basin.

## Files
- `calibrate_apr2026.py` — campaign driver (loader, scan set, anchors, merger). Run:
  `python apr2026/calibrate_apr2026.py` (full seedless pipeline) or `--quick` to refine
  from a seed. Writes `calibration_result.json` and the overlays here.
- `calibration_result.json` — the calibration below.
- `overlay_20260410-13_07_49.png` (high pitch), `..._13_54_52.png` (low pitch),
  `..._09_26_27.png` (roll 0.18°) — model-vs-data overlays.

## Result
| dP | dR | dY | dRP | dYP | Ecorr | disp (pivot 9046.82) |
|----|----|----|-----|-----|-------|------|
| 1.366 | 0.706 | +1.024 | +0.0449 | −0.0046 | −1.03 | 0.863 |

pass 0.99 (70/71 ridges < 6 eV), median 0.34 eV. Energy frame:
`E_corrected = 9046.82 + 0.862·(E_raw − 9046.82)` (true dE/px = 0.155), with
`E_raw = (pixel − 1550)·0.18 + offset`. offset = 9065 for the post-11:17 scans AND for
09_26_27 (its stated 9080 is a +15 eV spectrometer-GUI artifact — see below).

## Campaign-specific notes
- 6 scans: 3 high-pitch (66–77°), 2 low-pitch (34–41°), 1 different-roll (0.18° vs ~1.37°).
- Hand anchors (steep near-vertical lines, identified by eye): 35°=`[-1,1,1]`,
  36.6°=`[-1,-1,-1]`, 38°=`[-1,1,-1]`, 38.5°=`[-4,-2,2]`, 75.2°=`[-2,0,-2]`;
  09_26_27: 66.8°=`[-3,-1,1]`, 65.4°=`[-1,-3,1]`, 66.2°=`[-1,-1,-3]`;
  merger `[-3,-1,1]`/`[-1,-3,1]` at 66.1° (roll-1.37 scan only).
- **09_26_27 energy = 9065** (its 9080 is a +15 eV GUI artifact; on the −213 eV/deg
  `[-1,-1,-3]` line that 15 eV looked like a 0.07° pitch shift). This correction is what made
  the yaw determinable: **dY = +1.0° is a genuine, data-determined mount yaw** (not pinned).
- dispersion left fully open (free optimum 0.862 sits inside the ±2% `[-1,-1,5]` cone anyway).
- Ridge cleanup: per-ridge perpendicular outlier rejection (drops crossing-line stragglers)
  + `prune_anchors` (drops anchored ridges the geometry can't place - the anchor windows
  grabbed short crossing-line fragments in the crowded 66.8° near-vertical region). Both in
  the pipeline. 1 residual fail remains (a non-anchored shallow fragment @ 66.3°, cosmetic).
- The `[-3,-1,1]`/`[-1,-3,1]` identity is roll/yaw-dependent; no-swap confirmed (swap diverges).
