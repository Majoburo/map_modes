# EMRI One-Mode Map

Map regions of EMRI parameter space with [FastEMRIWaveforms (FEW)](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms) where one GW mode dominates (brightest vs 2nd brightest SNR), so a detector might see only that mode. Results are stored in an HDF5 file for later plotting or analysis.

---

## Quickstart

```bash
git clone --recurse-submodules git@github.com:Majoburo/map_modes.git
cd map_modes
uv sync
uv run python kerr_map.py # sch_map.py for schwarzchild.
```
⸻

What it does
- Defines parameter ranges for:
    - SMBH mass M1 (log10)
    - Compact object mass M2 (log10)
    - SMBH spin a
    - Initial eccentricity e0
    - Initial semi-latus rectum p0
    - Observer polar/azimuthal angles (θ, φ)
- Samples points uniformly using a Latin Hypercube sampler.
- Evaluates per-mode SNRs with FEW (Kerr, eccentric, equatorial inspirals).
- Keeps points where where one GW mode dominates; SNR_mode1/SNR_mode2 ≥ THR_RATIO.
- Saves points and their (l,m,k,n) mode indices into an HDF5 file.

⸻

Outputs
- snr_ratio_kerr.h5 — HDF5 file with:
- /pts: array of sampled parameters (log10_m1, log10_m2, a, p0, e0, theta, phi)
- /modes: corresponding single surviving mode indices (l, m, k, n)
- Attributes: DT_SEC, T_YEARS, THR_SNR, SEED, N_DONE

Each new run appends more samples to the same file.

⸻

Adjustable settings (top of kerr_map.py)
- Observation grid: DT_SEC, T_YEARS
- Threshold: THR_RATIO
- Number of samples: SCAN_SAMPLES
- Random seed: RANDOM_SEED
- Parameter ranges: LOG10_M1_RANGE, LOG10_M2_RANGE, a_RANGE, e0_RANGE, p0_RANGE, THETA_RANGE, PHI_RANGE
- Output file prefix: SAVE_PREFIX

⸻

Notes
- Resume is automatic: if snr_ratio_kerr.h5 exists, the script appends new points.
- If no one-mode points are found, try increasing SCAN_SAMPLES or lowering THR_SNR.

