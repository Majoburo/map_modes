import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib as mpl
import h5py

SAVE_PREFIX = "snr_ratio_kerr_keep_10"  # output prefix for HDF5 and PNG

def make_scatter_corner(pts, mode_indices):
    cols = ["log10_m1","log10_m2","a","p0","e0","theta","phi","snr_ratio"]
    vars_no_snr = ["log10_m1","log10_m2","a","p0","e0","theta","phi"]
    if pts.size == 0:
        raise ValueError("No points in HDF5 yet; run the mapper first.")

    df = pd.DataFrame(pts, columns=cols)
    #df['col_A'].mask(df['snr_ratio'] < 4, other=999)
    df = df[df['snr_ratio'] < 4]

    # Build a PairGrid excluding snr_ratio from the axes,
    # but keep it available for continuous hue coloring.
    g = sns.PairGrid(df, vars=vars_no_snr, corner=True, height=2.6, diag_sharey=False)
    
    hue_vals = df["snr_ratio"].to_numpy()
    vmin = max(hue_vals.min(), 0.0)
    vmax = hue_vals.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    g.map_lower(
        sns.scatterplot,
        s=3, alpha=1, linewidth=0, rasterized=True,
        hue=hue_vals, palette="magma", hue_norm=norm, legend=False
    )
    
    # diagonal histograms
    g.map_diag(sns.histplot, bins=52, fill=True, linewidth=0.0)

    # Single shared colorbar for continuous log-hue
    sm = mpl.cm.ScalarMappable(cmap="magma", norm=norm)
    sm.set_array([])
    # Attach colorbar to all axes in the grid
    valid_axes = [ax for ax in g.axes.flat if ax is not None]
    from matplotlib.ticker import LogLocator, FuncFormatter
    g.fig.colorbar(
        sm,
        ax=valid_axes,
        fraction=0.03,
        pad=0.04,
        ticks=LogLocator(base=10, subs=(1, 2, 5)),           # tick positions
        format=FuncFormatter(lambda v, pos: f"{v:.2g}")      # numeric labels
    )
    #g.fig.colorbar(sm, ax=valid_axes, fraction=0.03, pad=0.04, label="snr_ratio (log scale)")
    return g


def load_from_h5(path: str):
    """Return (pts, mode_indices). Missing file ⇒ empty arrays."""
    if not os.path.exists(path):
        return np.empty((0,7), float), np.empty((0,4), int)
    with h5py.File(path, "r") as f:
        pts = np.asarray(f["pts"]) if "pts" in f else np.empty((0,5), float)
        modes = np.asarray(f["modes"]) if "modes" in f else np.empty((0,4), int)
        return pts, modes

# ----------------------------
# Main
# ----------------------------

def main():
    h5_path = f"{SAVE_PREFIX}.h5"

    # Load existing points & state if present
    pts, mode_indices = load_from_h5(h5_path)
    if pts.size == 0:
        print("[plot] No data to plot — run the scan to generate HDF5 first.")
        return
    g = make_scatter_corner(pts, mode_indices)
    out_png = f"{SAVE_PREFIX}_corner_scatter.png"
    g.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[scan] Wrote corner plot to {out_png}")

if __name__ == "__main__":
    main()
