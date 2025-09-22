import math
import os
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import qmc
import pandas as pd
import seaborn as sns


from few import get_file_manager
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.waveform import FastSchwarzschildEccentricFlux

# ----------------------------
# Settings (edit these only)
# ----------------------------
# Observation / integration granularity — coarser values make FEW runs much faster.
DT_SEC = 100.0      # seconds per sample
T_YEARS = 0.1       # total duration in years
THR_SNR = 10.       # absolute per-mode SNR threshold (keep modes with SNR >= THR_SNR)
RANDOM_SEED = 123

# --- Mapping settings for 1-mode region ---
SCAN_SAMPLES = 2**int(np.log2(200_000))   # total random samples over the full prior hyper-rectangle

SAVE_PREFIX = "one_mode_map"  # output prefix for CSV and PNG

# --- Plotting & storage controls ---
# Shared plotting layout for corner
LABELS = [r"$\log_{10} M_1$", r"$\log_{10} M_2$", r"$e_0$", r"$p_0$"]

# Parameter ranges (intrinsic)
LOG10_M1_RANGE = (math.log10(5e5), math.log10(2e6))  # MBH mass
LOG10_M2_RANGE = (math.log10(1e1), math.log10(1e2))  # compact object mass
e0_RANGE = (0.0, 0.75)
U_RANGE = (0.0, 1.0)  # maps to p0 via feasibility: p0 = 10 + u*(6 + 2 e0)

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

    mode_selector_kwargs = {"sensitivity_fn": sens_fn, "dt": DT_SEC, "dist": DIST_GPC}
    few_nw = FastSchwarzschildEccentricFlux(
        inspiral_kwargs={"DENSE_STEPPING": 0, "buffer_length": int(1e3)},
        amplitude_kwargs={"buffer_length": int(1e3)},
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


def eval_mode(m1: float, m2: float, p0: float, e0: float, thr: float):
    """Call FEW once and return (num_kept, ls, ms, ks, ns).
    Arrays may be empty if no modes are kept.
    """
    few_nw = get_few()
    # Run the noise-weighted selection with absolute SNR threshold
    few_nw(
        float(m1), float(m2), float(p0), float(e0),
        THETA, PHI,
        T=T_YEARS,
        snr_abs_threshold=float(thr),
        dist=float(DIST_GPC),
        dt=float(DT_SEC),
    )
    n_kept = int(few_nw.num_modes_kept)
    ls = np.atleast_1d(few_nw.ls)
    ms = np.atleast_1d(few_nw.ms)
    ks = np.atleast_1d(few_nw.ks)
    ns = np.atleast_1d(few_nw.ns)
    return n_kept, ls, ms, ks, ns


def _p0_from_u(u: float, e0: float) -> float:
    return 10.0 + float(u) * (6.0 + 2.0 * float(e0))


def _sample_uniform(n, rng):
    sampler = qmc.Sobol(d=4, scramble=True, seed=rng.integers(2**31-1))
    X = sampler.random(n)
    l1 = LOG10_M1_RANGE[0] + X[:,0]*(LOG10_M1_RANGE[1]-LOG10_M1_RANGE[0])
    l2 = LOG10_M2_RANGE[0] + X[:,1]*(LOG10_M2_RANGE[1]-LOG10_M2_RANGE[0])
    u  = U_RANGE[0]        + X[:,2]*(U_RANGE[1]-U_RANGE[0])
    e0 = e0_RANGE[0]       + X[:,3]*(e0_RANGE[1]-e0_RANGE[0])
    return l1, l2, u, e0


def random_scan_one_mode(n_samples: int, seed: int = RANDOM_SEED):
    """
    Randomly sample the full parameter box and keep parameter tuples where the
    evaluator returns exactly one kept mode.

    Returns:
        pts:          float array (N, 5): [log10_m1, log10_m2, u, e0, p0]
        mode_indices: int array (N, 4): kept base mode (l,m,k,n) for each point
    """
    rng = np.random.default_rng(seed)
    keep = []
    modes_rec = []

    # First, uniform sampling over the full box
    l1, l2, uu, ee0 = _sample_uniform(n_samples, rng)
    for i in tqdm(range(n_samples), desc="[scan] uniform", total=n_samples, leave=False):
        lm1 = round(float(l1[i]), 6)
        lm2 = round(float(l2[i]), 6)
        u = round(float(uu[i]), 6)
        e0 = round(float(ee0[i]), 6)

        n, mode_tuple = eval_count_and_mode(lm1, lm2, u, e0, THR_SNR)
        if n == 1.0:
            p0 = _p0_from_u(u, e0)
            keep.append((lm1, lm2, e0, p0))
            modes_rec.append(mode_tuple)

    if not keep:
        return np.empty((0, 4), float), np.empty((0, 4), int)
    return np.array(keep, float), np.array(modes_rec, int)


def make_scatter_corner(pts, mode_indices):
    cols = ["log10_m1","log10_m2","e0","p0"]
    df = pd.DataFrame(pts, columns=cols)
    df["mode"] = [f"{l},{m},{k},{n}" for (l,m,k,n) in mode_indices]

    # keep legend small
    top = set(df["mode"].value_counts().index[:12])
    df["mode_plot"] = np.where(df["mode"].isin(top), df["mode"], "other")

    g = sns.PairGrid(df, vars=cols, hue="mode_plot", corner=True, height=2.6, diag_sharey=False)
    g.map_lower(sns.scatterplot, s=8, alpha=0.55, linewidth=0)
    g.map_diag(sns.histplot, bins=40, element="step", fill=False, linewidth=1.0)

    g.add_legend(frameon=False, title="mode", labelspacing=0.3, handlelength=0.8)
    return g


def eval_count_and_mode(
    log10_m1: float, log10_m2: float, u: float, e0: float, thr: float
) -> tuple[float, tuple[int, int, int, int]]:
    """
    Return (num_kept, (l,m,k,n)) where the tuple is the single kept base mode if num_kept==1,
    or (-1,-1,-1,-1) otherwise. Returns (+inf, (-1,-1,-1,-1)) on infeasible/error.
    """
    m1 = 10 ** float(log10_m1)
    m2 = 10 ** float(log10_m2)
    p0 = _p0_from_u(u, e0)

    # Feasibility guard
    if not (10.0 <= p0 <= 16.0 + 2.0 * e0) or not (0.0 <= e0 <= 0.75):
        return float("inf"), (-1, -1, -1, -1)
    # Ask FEW to provide number of kept modes and their (l,m,k,n)
    try:
        n_kept, ls, ms, ks, ns = eval_mode(m1, m2, p0, e0, thr)
    except Exception:
        return float("inf"), (-1, -1, -1, -1)
    
    if n_kept == 1 and len(ls) >= 1:
        mode_tuple = (int(ls[0]), int(ms[0]), int(ks[0]), int(ns[0]))
    else:
        mode_tuple = (-1, -1, -1, -1)
    return float(n_kept), mode_tuple


def save_pts_csv(path, pts, mode_indices):
    arr = np.concatenate([pts, mode_indices.astype(float)], axis=1)
    header = "log10_m1,log10_m2,e0,p0,l,m,k,n"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def load_pts_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    pts = np.stack([data["log10_m1"], data["log10_m2"], data["e0"], data["p0"]], axis=1).astype(float)
    mode_indices = np.stack([data["l"], data["m"], data["k"], data["n"]], axis=1).astype(int)
    return pts, mode_indices


def main():
    print(f"[diag] dt={DT_SEC:.1f}s, T={T_YEARS:.4f}yr → ~{int(T_YEARS*31557600/DT_SEC):,} samples/call")
    csv_path = f"{SAVE_PREFIX}.csv"
    if os.path.exists(csv_path):
        print(f"[replot] Using {csv_path}")
        pts, mode_indices = load_pts_csv(csv_path)
    else:
        print("[scan] Mapping the 1‑mode region …")
        pts, mode_indices = random_scan_one_mode(SCAN_SAMPLES, seed=RANDOM_SEED)
        if pts.size == 0:
            print("[scan] No 1‑mode points found. Increase SCAN_SAMPLES or lower THR_SNR.")
            return
        save_pts_csv(csv_path, pts, mode_indices)
        print(f"[scan] Saved {len(pts)} points → {csv_path}")

    fig = make_scatter_corner(pts, mode_indices)
    out_png = f"{SAVE_PREFIX}_corner_scatter.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[scan] Wrote corner plot to {out_png}")


if __name__ == "__main__":
    main()