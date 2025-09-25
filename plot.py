import os
import numpy as np
import pandas as pd
import seaborn as sns
import h5py

SAVE_PREFIX = "one_mode_map_kerr"  # output prefix for HDF5 and PNG

def make_scatter_corner(pts, mode_indices):
    cols = ["log10_m1","log10_m2","a","p0","e0","theta","phi"]
    if pts.size == 0:
        raise ValueError("No points in HDF5 yet; run the mapper first.")

    df = pd.DataFrame(pts, columns=cols)
    df["mode"] = [f"{l},{m},{k},{n}" for (l,m,k,n) in mode_indices]

    # keep legend small
    top = set(df["mode"].value_counts().index[:12])
    df["mode_plot"] = np.where(df["mode"].isin(top), df["mode"], "other")

    g = sns.PairGrid(df, vars=cols, hue="mode_plot", corner=True, height=2.6, diag_sharey=False)

    # base scatter
    g.map_lower(sns.scatterplot, s=1, alpha=0.15, linewidth=0, rasterized=True)
    # diagonal histograms
    g.map_diag(sns.histplot, bins=40, element="step", fill=False, linewidth=1.0)

    # tidy legend (robust across seaborn versions)
    g.add_legend(frameon=False, title="mode", labelspacing=0.3, handlelength=0.8, markerscale=8.0)
    legend = getattr(g, "_legend", None)
    for lh in legend.legend_handles:
        try:
            lh.set_alpha(0.9)
        except Exception:
            pass
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