import math
from functools import lru_cache
import os
import numpy as np
import corner as _corner
from tqdm.auto import tqdm as _tqdm
from scipy.stats import qmc
import matplotlib.pyplot as plt

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
SCAN_SAMPLES = 1_024    # total random samples over the full prior hyper-rectangle

SAVE_PREFIX = "one_mode_map"  # output prefix for CSV and PNG

# --- Plotting & storage controls ---
# Shared plotting layout for corner
SELECT_COLS = [0, 1, 3, 4]  # indices -> (log10_m1, log10_m2, e0, p0)
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
        n, mode_tuple = eval_count_and_mode_cached(lm1, lm2, u, e0, THR_SNR)
        if n == 1.0:
            p0 = _p0_from_u(u, e0)
            keep.append((lm1, lm2, u, e0, p0))
            modes_rec.append(mode_tuple)

    if not keep:
        return np.empty((0, 5), float), np.empty((0, 4), int)
    return np.array(keep, float), np.array(modes_rec, int)


def make_scatter_corner(
    pts: np.ndarray,
    mode_indices: np.ndarray,
):
    """
    Corner-style scatter plot colored by the kept base mode (l,m,k,n).
    pts columns: [log10_m1, log10_m2, u, e0, p0]
    """
    X = pts[:, SELECT_COLS]
    labels = LABELS
    alpha = 0.15
    ms = 6

    # Build categorical labels from (l,m,k,n)
    mode_labels = np.array([f"{l},{m},{k},{n}" for (l,m,k,n) in mode_indices], dtype=object)
    uniq = np.unique(mode_labels)
    # push unknown "-1,-1,-1,-1" to end
    uniq = np.array(sorted(uniq, key=lambda s: (s == "-1,-1,-1,-1", s)))

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    color_map = {lab: prop_cycle[i % len(prop_cycle)] for i, lab in enumerate(uniq)}

    fig = _corner.corner(
        X,
        labels=labels,
        bins=50,
        plot_datapoints=False,
        plot_contours=False,
        fill_contours=False,
        hist_bin_factor=2,
    )
    axes = np.array(fig.axes).reshape(len(labels), len(labels))

    # Off-diagonal scatter per category
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i > j:
                ax = axes[i, j]
                for lab in uniq:
                    mask = mode_labels == lab
                    if not np.any(mask):
                        continue
                    ax.scatter(
                        X[mask, j], X[mask, i],
                        s=ms, alpha=alpha, color=color_map[lab], rasterized=True
                    )

    # Simple legend (cap to 20 entries to avoid clutter)
    legend_ax = axes[0, -1]
    handles, labels_ = [], []
    for lab in uniq[:20]:
        handles.append(plt.Line2D([], [], marker='o', linestyle='None', markersize=6, color=color_map[lab], label=f"mode ({lab})" if lab != "-1,-1,-1,-1" else "mode ?"))
        labels_.append(f"mode ({lab})" if lab != "-1,-1,-1,-1" else "mode ?")
    legend_ax.legend(handles, labels_, loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


@lru_cache(maxsize=20000)
def eval_count_and_mode_cached(
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
    try:
        breakpoint()
        # Ask FEW to also provide per-mode SNR and base (l,m,k,n) mapping
        ret = few_noise_weighted(
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
            return_mode_snr=True,
        )
        
        # (mode_snr, l_base, m_base, k_base, n_base)
        n_kept = float(getattr(few_noise_weighted, "num_modes_kept", float("inf")))
        mode_tuple = (-1, -1, -1, -1)
        if isinstance(ret, tuple) and len(ret) >= 5:
            mode_snr, l_base, m_base, k_base, n_base = ret[-5:]
            try:
                import numpy as _np
                mask = _np.asarray(mode_snr) >= float(thr)
                if int(mask.sum()) == 1 and n_kept == 1.0:
                    idx = int(_np.where(mask)[0][0])
                    mode_tuple = (int(l_base[idx]), int(m_base[idx]), int(k_base[idx]), int(n_base[idx]))
            except Exception:
                pass
        return n_kept, mode_tuple
    except Exception as e:
        print(f"[eval-error] {e} @ (log10_m1={log10_m1}, log10_m2={log10_m2}, u={u}, e0={e0})")
        return float("inf"), (-1, -1, -1, -1)


def save_pts_csv(path, pts, mode_indices):
    arr = np.concatenate([pts, mode_indices.astype(float)], axis=1)
    header = "log10_m1,log10_m2,u,e0,p0,l,m,k,n"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")

def load_pts_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    get = lambda k: np.asarray(data[k])
    pts = np.stack([get("log10_m1"), get("log10_m2"), get("u"), get("e0"), get("p0")], axis=1).astype(float)
    mode_indices = np.stack([get("l"), get("m"), get("k"), get("n")], axis=1).astype(int)
    return pts, mode_indices


def main():
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