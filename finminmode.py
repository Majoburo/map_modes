import math
from functools import lru_cache
import os
import numpy as np
import corner as _corner
from tqdm.auto import tqdm as _tqdm
from scipy.stats import qmc

from few import get_file_manager
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.waveform import FastSchwarzschildEccentricFlux

# ----------------------------
# Settings (edit these only)
# ----------------------------
# Observation / integration granularity — coarser values make FEW runs much faster.
DT_SEC = 100.0         # seconds per sample
T_YEARS = 0.1       # total duration in years
THR_SNR = 17.       # absolute per-mode SNR threshold (keep modes with SNR >= THR_SNR)
RANDOM_SEED = 123

# --- Mapping settings for 1-mode region ---
SCAN_SAMPLES = 2**int(np.log2(200000))    # total random samples over the full prior hyper-rectangle
SCAN_REFINE_ROUNDS = 2  # how many local refinement rounds around 1-mode hits
SCAN_REFINE_FRACTION = 0.25  # fraction of current hits to jitter-refine per round
SCAN_JITTER_SIGMA = np.array([0.05, 0.05, 0.05, 0.05])  # std-dev for (log10_m1, log10_m2, u, e0)
SAVE_PREFIX = "one_mode_map"  # output prefix for CSV and PNG

# --- Plotting & storage controls ---
# If True and a CSV exists, skip scanning and just replot from disk.
REUSE_EXISTING = True
# Use only uniform samples (no-refine) to drive corner KDE/histograms.
PLOT_UNIFORM_ONLY_DENSITY = True
# If refine points are included in KDE, this is their relative weight.
REFINE_KDE_WEIGHT = 0.1
# Overlay transparency for refine points in off-diagonal panels.
REFINE_OVERLAY_ALPHA = 0.

# Parameter ranges (intrinsic)
LOG10_M1_RANGE = (math.log10(5e5), math.log10(2e6))  # MBH mass
LOG10_M2_RANGE = (math.log10(1e1), math.log10(1e2))  # compact object mass
e0_RANGE = (0.0, 0.75)
U_RANGE = (0.0, 1.0)  # maps to p0 via feasibility: p0 = 10 + u*(6 + 2 e0)

# Fixed angles / distance (edit if desired)
THETA = np.pi / 3.0
PHI = np.pi / 4.0
DIST_GPC = 1.0

# ----------------------------
# FEW setup
# ----------------------------
noise = np.genfromtxt(get_file_manager().get_file("LPA.txt"), names=True)
f = np.asarray(noise["f"], dtype=np.float64)
PSD = np.asarray(noise["ASD"], dtype=np.float64) ** 2


sens_fn = CubicSplineInterpolant(f, PSD)
class ClippedInterpolant:
    def __init__(self, base):
        self.base = base
        # FEW’s CubicSplineInterpolant stores t as shape (1, N)
        self._lo = float(np.ravel(base.t)[0])
        self._hi = float(np.ravel(base.t)[-1])
    def __call__(self, x):
        x = np.asarray(x)
        return self.base(np.clip(x, self._lo, self._hi))

sens_fn = ClippedInterpolant(sens_fn)  # after you create the base interpolant

inspiral_kwargs = {"DENSE_STEPPING": 0, "buffer_length": int(1e3)}
amplitude_kwargs = {"buffer_length": int(1e3)}
Ylm_kwargs = {"include_minus_m": False}
sum_kwargs = {"pad_output": False}
mode_selector_kwargs = {"sensitivity_fn": sens_fn, "dt": DT_SEC, "dist": DIST_GPC }

few_noise_weighted = FastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    mode_selector_kwargs=mode_selector_kwargs,
)

print(
    f"[diag] dt={DT_SEC:.1f}s, T={T_YEARS:.4f}yr → ~{int(T_YEARS*31557600/DT_SEC):,} samples/call"
)

# ----------------------------
# Black-box evaluator with caching
# ----------------------------

def _p0_from_u(u: float, e0: float) -> float:
    return 10.0 + float(u) * (6.0 + 2.0 * float(e0))


def _clip_params(log10_m1, log10_m2, u, e0):
    log10_m1 = float(np.clip(log10_m1, *LOG10_M1_RANGE))
    log10_m2 = float(np.clip(log10_m2, *LOG10_M2_RANGE))
    u = float(np.clip(u, *U_RANGE))
    e0 = float(np.clip(e0, *e0_RANGE))
    return log10_m1, log10_m2, u, e0


def _sample_uniform(n, rng):
    sampler = qmc.Sobol(d=4, scramble=True, seed=rng.integers(2**31-1))
    X = sampler.random(n)
    l1 = LOG10_M1_RANGE[0] + X[:,0]*(LOG10_M1_RANGE[1]-LOG10_M1_RANGE[0])
    l2 = LOG10_M2_RANGE[0] + X[:,1]*(LOG10_M2_RANGE[1]-LOG10_M2_RANGE[0])
    u  = U_RANGE[0]        + X[:,2]*(U_RANGE[1]-U_RANGE[0])
    e0 = e0_RANGE[0]       + X[:,3]*(e0_RANGE[1]-e0_RANGE[0])
    return l1, l2, u, e0


def random_scan_one_mode(n_samples: int, seed: int = RANDOM_SEED, refine_rounds: int = 1):
    """
    Randomly sample the full parameter box and keep parameter tuples where the
    evaluator returns exactly one kept mode. Then optionally refine locally around
    a subset of those hits by jittering the parameters.

    Returns:
        pts:      float array of shape (N, 7) with columns [log10_m1, log10_m2, u, e0, p0, m1, m2]
        weights:  float array of shape (N,) with plotting weights (1.0 for uniform, REFINE_KDE_WEIGHT for refine)
        is_refine:bool array of shape (N,) True for refine-origin points, False for uniform
    """
    rng = np.random.default_rng(seed)
    keep = []
    keep_w = []
    is_ref = []
    seen = set()

    # First, uniform sampling over the full box
    l1, l2, uu, ee0 = _sample_uniform(n_samples, rng)
    for i in _tqdm(range(n_samples), desc="[scan] uniform", total=n_samples, leave=False):
        lm1 = round(float(l1[i]), 6)
        lm2 = round(float(l2[i]), 6)
        u = round(float(uu[i]), 6)
        e0 = round(float(ee0[i]), 6)
        key = (lm1, lm2, u, e0)
        if key in seen:
            continue
        seen.add(key)
        n = eval_count_cached(lm1, lm2, u, e0, THR_SNR)
        if n == 1.0:
            p0 = _p0_from_u(u, e0)
            keep.append((lm1, lm2, u, e0, p0))
            keep_w.append(1.0)
            is_ref.append(False)

    # Local refinements around the discovered hits (helps trace the boundary)
    for r in range(refine_rounds):
        if not keep:
            break
        # choose a subset to jitter
        idx = rng.choice(len(keep), size=max(1, int(len(keep) * SCAN_REFINE_FRACTION)), replace=False)
        seeds = np.array([keep[i][:4] for i in idx])  # columns: log10_m1, log10_m2, u, e0
        noise = rng.normal(loc=0.0, scale=SCAN_JITTER_SIGMA, size=seeds.shape)
        proposals = seeds + noise
        for row in _tqdm(proposals, desc=f"[scan] refine {r+1}/{refine_rounds}", leave=False):
            lm1, lm2, u, e0 = _clip_params(*row)
            lm1 = round(lm1, 6); lm2 = round(lm2, 6); u = round(u, 6); e0 = round(e0, 6)
            key = (lm1, lm2, u, e0)
            if key in seen:
                continue
            seen.add(key)
            n = eval_count_cached(lm1, lm2, u, e0, THR_SNR)
            if n == 1.0:
                p0 = _p0_from_u(u, e0)
                keep.append((lm1, lm2, u, e0, p0))
                keep_w.append(REFINE_KDE_WEIGHT)
                is_ref.append(True)

    if not keep:
        return np.empty((0, 5), dtype=np.float64), np.empty((0,), dtype=np.float64), np.zeros((0,), dtype=bool)
    return np.array(keep, dtype=np.float64), np.array(keep_w, dtype=np.float64), np.array(is_ref, dtype=bool)


def make_corner_plot(
    pts: np.ndarray,
    is_refine: np.ndarray = None,
    weights: np.ndarray = None,
    uniform_only_density: bool = PLOT_UNIFORM_ONLY_DENSITY,
    refine_weight: float = REFINE_KDE_WEIGHT,
    overlay_alpha: float = REFINE_OVERLAY_ALPHA,
):
    """
    Corner-style plot of the 1-mode region using variables:
    (log10_m1, log10_m2, e0, p0).
    Density (KDE + histograms) is built from uniform-only by default to avoid boundary bias.
    Refine points are overlaid as faint scatter on off-diagonals.

    pts columns: [log10_m1, log10_m2, u, e0, p0]
    """
    X = pts[:, [0, 1, 3, 4]]
    labels = [r"$\log_{10} M_1$", r"$\log_{10} M_2$", r"$e_0$", r"$p_0$"]

    if is_refine is None:
        # If no provenance provided, assume all uniform
        is_refine = np.zeros(len(pts), dtype=bool)

    # Build plotting weights: zero out refine for density if requested
    if weights is None:
        weights = np.ones(len(pts), dtype=float)
    if uniform_only_density:
        kde_weights = weights.copy()
        kde_weights[is_refine] = 0.0
    else:
        kde_weights = weights.copy()
        kde_weights[is_refine] = float(refine_weight)

    fig = _corner.corner(
        X,
        labels=labels,
        weights=kde_weights,
        bins=50,
        #smooth=1.5,
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.50, 0.84, 0.97),
        hist_bin_factor=2,
    )

    # Overlay refine points as faint scatter in off-diagonals
    X_ref = X[is_refine]
    if X_ref.size:
        axes = np.array(fig.axes).reshape(len(labels), len(labels))
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i > j:
                    ax = axes[i, j]
                    ax.scatter(
                        X_ref[:, j], X_ref[:, i],
                        s=4, alpha=overlay_alpha, rasterized=True
                    )
    fig.tight_layout()
    return fig


@lru_cache(maxsize=20000)
def eval_count_cached(
    log10_m1: float, log10_m2: float, u: float, e0: float, thr: float
) -> float:
    """Return number of kept modes (>=0), or +inf on error. Cached by rounded args."""
    m1 = 10 ** float(log10_m1)
    m2 = 10 ** float(log10_m2)
    p0 = _p0_from_u(u, e0)


    # Feasibility guaranteed by construction, but guard anyway
    if not (10.0 <= p0 <= 16.0 + 2.0 * e0) or not (0.0 <= e0 <= 0.75):
        return float("inf")
    try:
        few_noise_weighted(
            m1,
            m2,
            p0,
            float(e0),
            THETA,
            PHI,
            T=T_YEARS,
            snr_abs_threshold=float(thr),
            dist=float(DIST_GPC),
            dt=float(DT_SEC),
        )
        return float(few_noise_weighted.num_modes_kept)
    except Exception as e:
        print(e)
        exit()
        return float("inf")


def save_pts_csv(path, pts, weights=None, is_refine=None):
    if weights is None:
        weights = np.ones(len(pts), float)
    if is_refine is None:
        is_refine = np.zeros(len(pts), bool)
    arr = np.concatenate([pts, is_refine.astype(int).reshape(-1,1), weights.reshape(-1,1)], axis=1)
    header = "log10_m1,log10_m2,u,e0,p0,is_refine,w"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")

def load_pts_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    # Support both structured and plain arrays
    cols = data.dtype.names
    get = lambda k: np.asarray(data[k])
    pts = np.stack([get("log10_m1"), get("log10_m2"), get("u"), get("e0"), get("p0")], axis=1)
    is_refine = get("is_refine").astype(int) if "is_refine" in cols else np.zeros(len(pts), int)
    weights = get("w") if "w" in cols else np.ones(len(pts), float)
    return pts.astype(float), weights.astype(float), is_refine.astype(bool)


def main():
    csv_path = f"{SAVE_PREFIX}.csv"
    if REUSE_EXISTING and os.path.exists(csv_path):
        print(f"[replot] Loading points from {csv_path} (skipping evaluation)...")
        pts, weights, is_refine = load_pts_csv(csv_path)
    else:
        print("\n[scan] Mapping the region where exactly one mode is kept (n == 1) ...")
        pts, weights, is_refine = random_scan_one_mode(SCAN_SAMPLES, seed=RANDOM_SEED, refine_rounds=SCAN_REFINE_ROUNDS)
        if pts.size == 0:
            print("[scan] No 1-mode points found. Try increasing SCAN_SAMPLES or relaxing THR_SNR.")
            return
        save_pts_csv(csv_path, pts, weights=weights, is_refine=is_refine)
        print(f"[scan] Saved {len(pts)} 1-mode points to {csv_path}")

    fig = make_corner_plot(
        pts,
        is_refine=is_refine,
        weights=weights,
        uniform_only_density=PLOT_UNIFORM_ONLY_DENSITY,
        refine_weight=REFINE_KDE_WEIGHT,
        overlay_alpha=REFINE_OVERLAY_ALPHA,
    )
    out_png = f"{SAVE_PREFIX}_corner.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[scan] Wrote corner plot to {out_png}")


if __name__ == "__main__":
    main()