import os
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import qmc
import h5py

from few import get_file_manager
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.waveform import FastKerrEccentricEquatorialFlux
from few.utils.constants import MRSUN_SI, Gpc

# ----------------------------
# Settings (edit these only)
# ----------------------------
# Observation / integration granularity — coarser values make FEW runs much faster.
DT_SEC = 10.0      # seconds per sample
T_YEARS = 0.1       # total duration in years
THR_SNR = 17.       # absolute per-mode SNR threshold (keep modes with SNR >= THR_SNR)
RANDOM_SEED = 123

# --- Mapping settings for 1-mode region ---
SCAN_SAMPLES = 2**int(np.log2(2_000_000))   # total random samples over the full prior hyper-rectangle, need to be a power of two for Sobol to work well

SAVE_PREFIX = "one_mode_map_kerr"  # output prefix for HDF5 and PNG

# --- Plotting & storage controls ---
# Parameter ranges (intrinsic)
LOG10_M1_RANGE = (np.log10(5e5), np.log10(2e6))  # MBH mass
LOG10_M2_RANGE = (np.log10(1e1), np.log10(1e2))  # compact object mass
a_RANGE = (0.0, 0.999)
e0_RANGE = (0.0, 0.75)
xI = 1 # Prograde orbits
p0_RANGE = (7.5, 17)

# Fixed angles / distance (edit if desired)
THETA = np.pi / 3.0
PHI = np.pi / 4.0
DIST_GPC = 1.0

class ClippedInterpolant:
    def __init__(self, base):
        self.base = base
        # FEW’s CubicSplineInterpolant stores t as shape (1, N)
        self._lo = float(np.ravel(base.t)[0])
        self._hi = float(np.ravel(base.t)[-1])
    def __call__(self, x):
        x = np.asarray(x)
        return self.base(np.clip(x, self._lo, self._hi))

def build_few():
    """Create and return the FEW object and any derived helpers."""
    noise = np.genfromtxt(get_file_manager().get_file("LPA.txt"), names=True)
    f = np.asarray(noise["f"], float)
    PSD = np.asarray(noise["ASD"], float)**2
    sens_fn = ClippedInterpolant(CubicSplineInterpolant(f, PSD))

    mode_selector_kwargs = {"sensitivity_fn": sens_fn}

    few_nw = FastKerrEccentricEquatorialFlux(
        inspiral_kwargs={"DENSE_STEPPING": 0, "buffer_length": int(1e3)},
        #amplitude_kwargs={"buffer_length": int(1e3)},
        Ylm_kwargs={"include_minus_m": False},
        sum_kwargs={"pad_output": False},
        mode_selector_kwargs=mode_selector_kwargs,
    )
    return few_nw

# Lazy singleton FEW instance so imports don't do heavy work
_FEW = None

def get_few():
    global _FEW
    if _FEW is None:
        _FEW = build_few()
    return _FEW


def eval_mode(m1: float, m2: float, a: float,  p0: float, e0: float, thr: float):
    """Call FEW once and return (num_kept, ls, ms, ks, ns).
    Arrays may be empty if no modes are kept.
    """
    few_nw = get_few()
    # Run the noise-weighted selection with absolute SNR threshold
    mu = m1 * m2 / (m1 + m2)
    dist_dimensionless = (DIST_GPC * Gpc) / (mu * MRSUN_SI)
    few_nw(
        m1, m2, a, p0, e0, xI,
        THETA, PHI,
        T=T_YEARS,
        dist=float(DIST_GPC),
        dt=float(DT_SEC),
        snr_abs_thr=thr*dist_dimensionless, # the SNR threshold is at source, multiply by distance for realistic SNR
    )
    n_kept = int(few_nw.num_modes_kept)
    ls = np.atleast_1d(few_nw.ls)
    ms = np.atleast_1d(few_nw.ms)
    ks = np.atleast_1d(few_nw.ks)
    ns = np.atleast_1d(few_nw.ns)
    return n_kept, ls, ms, ks, ns


def _sample_uniform(n, sobol_seed, n_skip=0):
    sampler = qmc.Sobol(d=5, scramble=True, seed=int(sobol_seed))
    if n_skip:
        sampler.fast_forward(int(n_skip))
    X = sampler.random(n)
    l1 = LOG10_M1_RANGE[0] + X[:,0]*(LOG10_M1_RANGE[1]-LOG10_M1_RANGE[0])
    l2 = LOG10_M2_RANGE[0] + X[:,1]*(LOG10_M2_RANGE[1]-LOG10_M2_RANGE[0])
    a  = a_RANGE[0]        + X[:,2]*(a_RANGE[1]-a_RANGE[0])
    p0 = p0_RANGE[0]       + X[:,3]*(p0_RANGE[1]-p0_RANGE[0])
    e0 = e0_RANGE[0]       + X[:,4]*(e0_RANGE[1]-e0_RANGE[0])
    return l1, l2, a, p0, e0


def random_scan_one_mode(n_samples: int, sobol_seed: int, sobol_n_skip: int):
    """
    Randomly sample the full parameter box and keep parameter tuples where the
    evaluator returns exactly one kept mode.

    Returns:
        pts:          float array (N, 5): [log10_m1, log10_m2, a, p0, e0]
        mode_indices: int array (N, 4): kept base mode (l,m,k,n) for each point
    """
    keep = []
    modes_rec = []
    l1,l2,aa,pp0,ee0 = _sample_uniform(n_samples, sobol_seed, n_skip=sobol_n_skip)
    for i in (pbar := tqdm(range(n_samples), desc="[scan] uniform", total=n_samples, leave=False)):
        lm1 = round(float(l1[i]), 6)
        lm2 = round(float(l2[i]), 6)
        a = round(float(aa[i]), 6)
        p0 = round(float(pp0[i]), 6)
        e0 = round(float(ee0[i]), 6)
        try:
            n, mode_tuple = eval_count_and_mode(lm1, lm2, a, p0, e0, THR_SNR)
        except Exception as e:
            print(e)
            continue
        if n == 1:
            keep.append((lm1, lm2, a, p0, e0)
                        )
            modes_rec.append(mode_tuple)
        pbar.set_postfix(kept=len(keep))
    if not keep:
        return np.empty((0, 5), float), np.empty((0, 4), int), sobol_seed, sobol_n_skip + n_samples
    return np.array(keep, float), np.array(modes_rec, int), sobol_seed, sobol_n_skip + n_samples


def eval_count_and_mode(
        log10_m1: float, log10_m2: float, a: float, p0: float, e0: float, thr: float
) -> tuple[int, tuple[int, int, int, int]]:
    """
    Return (num_kept, (l,m,k,n)) where the tuple is the single kept base mode if num_kept==1,
    or (-1,-1,-1,-1) otherwise. Returns (-1, (-1,-1,-1,-1)) on infeasible/error.
    """
    m1 = 10 ** float(log10_m1)
    m2 = 10 ** float(log10_m2)
    # Ask FEW to provide number of kept modes and their (l,m,k,n)
    try:
        n_kept, ls, ms, ks, ns = eval_mode(m1, m2, a, p0, e0, thr)
    except Exception as e:
        print(e)
        return -1, (-1, -1, -1, -1)

    if n_kept == 1 and len(ls) >= 1:
        mode_tuple = (int(ls[0]), int(ms[0]), int(ks[0]), int(ns[0]))
    else:
        mode_tuple = (-1, -1, -1, -1)
    return int(n_kept), mode_tuple


# ----------------------------
# HDF5 utilities
# ----------------------------

def _ensure_dset(f: h5py.File, name: str, shape, dtype, maxshape=(None,)):
    if name in f:
        return f[name]
    # Choose a chunk size that is friendly to append (powers of two rows)
    chunks = (max(1, min(4096, 1024)),) + tuple(shape[1:])
    return f.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype, chunks=chunks, compression="lzf")


def _append_rows(dset: h5py.Dataset, rows: np.ndarray):
    if rows.size == 0:
        return
    n_old = dset.shape[0]
    n_new = rows.shape[0]
    dset.resize((n_old + n_new,) + dset.shape[1:])
    dset[n_old:n_old + n_new, ...] = rows


def save_to_h5(path: str, pts: np.ndarray, mode_indices: np.ndarray, sobol_seed: int, sobol_n_done: int):
    """Append new rows to HDF5 and update run state as attributes."""
    with h5py.File(path, "a") as f:
        # Datasets live at root for simplicity
        pts_ds = _ensure_dset(f, "pts", shape=(0, 5), dtype="f8", maxshape=(None, 5))
        modes_ds = _ensure_dset(f, "modes", shape=(0, 4), dtype="i4", maxshape=(None, 4))
        # Append
        _append_rows(pts_ds, np.asarray(pts, dtype=np.float64))
        _append_rows(modes_ds, np.asarray(mode_indices, dtype=np.int32))
        # Metadata/state
        f.attrs["SOBOL_SEED"] = int(sobol_seed)
        f.attrs["SOBOL_N_DONE"] = int(sobol_n_done)
        # Helpful provenance
        f.attrs["DT_SEC"] = float(DT_SEC)
        f.attrs["T_YEARS"] = float(T_YEARS)
        f.attrs["THR_SNR"] = float(THR_SNR)
        f.attrs["columns_pts"] = np.array([b"log10_m1", b"log10_m2", b"a", b"p0", b"e0"], dtype="S")
        f.attrs["columns_modes"] = np.array([b"l", b"m", b"k", b"n"], dtype="S")


def load_from_h5(path: str):
    """Return (pts, mode_indices, sobol_seed, sobol_n_done). Missing file ⇒ empty arrays and default seed/done."""
    if not os.path.exists(path):
        return np.empty((0,5), float), np.empty((0,4), int), RANDOM_SEED, 0
    with h5py.File(path, "r") as f:
        pts = np.asarray(f["pts"]) if "pts" in f else np.empty((0,5), float)
        modes = np.asarray(f["modes"]) if "modes" in f else np.empty((0,4), int)
        sobol_seed = int(f.attrs.get("SOBOL_SEED", RANDOM_SEED))
        sobol_n_done = int(f.attrs.get("SOBOL_N_DONE", 0))
        return pts, modes, sobol_seed, sobol_n_done


# ----------------------------
# Main
# ----------------------------

def main():
    print(f"[diag] dt={DT_SEC:.1f}s, T={T_YEARS:.4f}yr → ~{int(T_YEARS*31557600/DT_SEC):,} samples/call")
    h5_path = f"{SAVE_PREFIX}.h5"

    # Load existing points & state if present
    pts, mode_indices, sobol_seed, sobol_n_done = load_from_h5(h5_path)

    if pts.size == 0:
        print("[scan] Mapping the 1‑mode region …")
        newpts, newmode_indices, sobol_seed, sobol_n_done = random_scan_one_mode(SCAN_SAMPLES, RANDOM_SEED, 0)
        if newpts.size == 0:
            print("[scan] No 1‑mode points found. Increase SCAN_SAMPLES or lower THR_SNR.")
            return
        save_to_h5(h5_path, newpts, newmode_indices, sobol_seed, sobol_n_done)
        pts = newpts
        mode_indices = newmode_indices
    else:
        print(f"[replot] Using {h5_path}")
        newpts, newmode_indices, sobol_seed, sobol_n_done = random_scan_one_mode(SCAN_SAMPLES, sobol_seed, sobol_n_done)
        if newpts.size > 0:
            save_to_h5(h5_path, newpts, newmode_indices, sobol_seed, sobol_n_done)
            # Concatenate for plotting this session (avoid full reload for speed)
            pts = np.concatenate([pts, newpts], axis=0)
            mode_indices = np.concatenate([mode_indices, newmode_indices], axis=0)
        else:
            # Still update state even if nothing appended (e.g., errors)
            save_to_h5(h5_path, np.empty((0,5)), np.empty((0,4)), sobol_seed, sobol_n_done)

    print(f"[scan] Saved {len(pts)} total points → {h5_path}")

if __name__ == "__main__":
    main()
