import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "output" / "results"
FIG_DIR = ROOT / "output" / "figures"

expr_norm = pd.read_csv(RES_DIR / "expr_processed.csv", index_col=0)
meta      = pd.read_csv(RES_DIR / "sample_metadata.csv", index_col=0)

tumor_cols    = meta[meta["label"] == 1].index.tolist()
nontumor_cols = meta[meta["label"] == 0].index.tolist()

print(f"Loaded: {expr_norm.shape[0]} probes x {expr_norm.shape[1]} samples")
print(f"Tumor: {len(tumor_cols)}, Non-tumor: {len(nontumor_cols)}")

print("\nRunning Welch's t-test per probe...")

tumor_data    = expr_norm[tumor_cols].values
nontumor_data = expr_norm[nontumor_cols].values

t_stats, p_vals = stats.ttest_ind(tumor_data, nontumor_data, axis=1, equal_var=False)

mean_tumor    = tumor_data.mean(axis=1)
mean_nontumor = nontumor_data.mean(axis=1)
log2fc        = mean_tumor - mean_nontumor  # data is already in log2 space

_, adj_p, _, _ = multipletests(p_vals, method="fdr_bh")

deg_stats = pd.DataFrame({
    "probe_id":      expr_norm.index,
    "log2FC":        log2fc,
    "p_value":       p_vals,
    "adj_p_value":   adj_p,
    "mean_tumor":    mean_tumor,
    "mean_nontumor": mean_nontumor,
}, index=expr_norm.index)

FC_THRESH = 1.0
P_THRESH  = 0.05

sig_mask  = (deg_stats["adj_p_value"] < P_THRESH) & (deg_stats["log2FC"].abs() >= FC_THRESH)
degs      = deg_stats[sig_mask].copy()
up_degs   = degs[degs["log2FC"] > 0]
down_degs = degs[degs["log2FC"] < 0]

print(f"\nDEG Results (adj.p < {P_THRESH}, |log2FC| >= {FC_THRESH}):")
print(f"  Total DEGs      : {len(degs)}")
print(f"  Upregulated     : {len(up_degs)}")
print(f"  Downregulated   : {len(down_degs)}")
print(f"  Non-significant : {len(deg_stats) - len(degs)}")

deg_stats.sort_values("adj_p_value").to_csv(RES_DIR / "deg_all_probes.csv")
degs.sort_values("log2FC", ascending=False).to_csv(RES_DIR / "deg_significant.csv")
print(f"\nSaved DEG tables -> {RES_DIR}")

def sig_label(row):
    if row["adj_p_value"] < P_THRESH and row["log2FC"] >= FC_THRESH:
        return "Up"
    elif row["adj_p_value"] < P_THRESH and row["log2FC"] <= -FC_THRESH:
        return "Down"
    return "NS"

deg_stats["group"] = deg_stats.apply(sig_label, axis=1)
color_map = {"Up": "#c0392b", "Down": "#2980b9", "NS": "#bdc3c7"}

legend_labels = {
    "Up":   f"Upregulated ({len(up_degs)})",
    "Down": f"Downregulated ({len(down_degs)})",
    "NS":   f"Non-significant ({len(deg_stats)-len(degs)})",
}

print("Generating volcano plot...")
fig, ax = plt.subplots(figsize=(8, 7))
for grp, df_grp in deg_stats.groupby("group"):
    ax.scatter(df_grp["log2FC"], -np.log10(df_grp["adj_p_value"]),
               c=color_map[grp], s=6, alpha=0.5, label=grp, rasterized=True)
ax.axvline(x=FC_THRESH,  color="gray", linestyle="--", linewidth=0.8)
ax.axvline(x=-FC_THRESH, color="gray", linestyle="--", linewidth=0.8)
ax.axhline(y=-np.log10(P_THRESH), color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("log2 Fold Change (Tumor vs Non-tumor)", fontsize=12)
ax.set_ylabel("-log10(adj. p-value)", fontsize=12)
ax.set_title("Volcano Plot: Differentially Expressed Probes", fontsize=13)
handles = [plt.scatter([], [], c=color_map[g], s=30, label=legend_labels[g])
           for g in ["Up", "Down", "NS"]]
ax.legend(handles=handles, fontsize=10)
plt.tight_layout()
fig.savefig(FIG_DIR / "02a_volcano_plot.png", dpi=300)
plt.close()
print("  Saved -> 02a_volcano_plot.png")

print("Generating MA plot...")
deg_stats["A"] = (deg_stats["mean_tumor"] + deg_stats["mean_nontumor"]) / 2
fig, ax = plt.subplots(figsize=(8, 6))
for grp, df_grp in deg_stats.groupby("group"):
    ax.scatter(df_grp["A"], df_grp["log2FC"],
               c=color_map[grp], s=5, alpha=0.4, label=grp, rasterized=True)
ax.axhline(y=0,          color="black", linewidth=0.8)
ax.axhline(y=FC_THRESH,  color="gray",  linestyle="--", linewidth=0.8)
ax.axhline(y=-FC_THRESH, color="gray",  linestyle="--", linewidth=0.8)
ax.set_xlabel("A = (Mean Tumor + Mean Non-tumor) / 2", fontsize=11)
ax.set_ylabel("M = log2 Fold Change", fontsize=11)
ax.set_title("MA Plot: Mean Expression vs Fold Change", fontsize=13)
handles = [plt.scatter([], [], c=color_map[g], s=30, label=legend_labels[g])
           for g in ["Up", "Down", "NS"]]
ax.legend(handles=handles, fontsize=10)
plt.tight_layout()
fig.savefig(FIG_DIR / "02b_ma_plot.png", dpi=300)
plt.close()
print("  Saved -> 02b_ma_plot.png")

print("Generating top DEG barplot...")
top_degs = degs.reindex(degs["log2FC"].abs().sort_values(ascending=False).index).head(30)
top_degs = top_degs.sort_values("log2FC")
colors = ["#2980b9" if v < 0 else "#c0392b" for v in top_degs["log2FC"]]
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(range(len(top_degs)), top_degs["log2FC"], color=colors, edgecolor="none")
ax.set_yticks(range(len(top_degs)))
ax.set_yticklabels(top_degs["probe_id"].astype(str), fontsize=8)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("log2 Fold Change", fontsize=12)
ax.set_title("Top 30 DEGs by |log2FC| (Tumor vs Non-tumor)", fontsize=12)
ax.set_xlim(top_degs["log2FC"].min() - 0.5, top_degs["log2FC"].max() + 0.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#c0392b", label="Upregulated"),
                   Patch(color="#2980b9", label="Downregulated")], fontsize=10)
plt.tight_layout()
fig.savefig(FIG_DIR / "02c_top30_deg_barplot.png", dpi=300)
plt.close()
print("  Saved -> 02c_top30_deg_barplot.png")

print("Generating heatmap of top 50 DEGs...")
top50     = degs.reindex(degs["log2FC"].abs().sort_values(ascending=False).index).head(50)
heat_data = expr_norm.loc[top50.index, tumor_cols + nontumor_cols]
heat_z    = heat_data.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
col_colors = ["#c0392b"] * len(tumor_cols) + ["#2980b9"] * len(nontumor_cols)
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(heat_z, cmap="RdBu_r", center=0, ax=ax,
            xticklabels=False, yticklabels=False,
            cbar_kws={"label": "Z-score"})
ax.set_title("Heatmap: Top 50 DEGs (Z-score normalized)", fontsize=13)
ax.set_xlabel("Samples  [Tumor (red) | Non-tumor (blue)]", fontsize=10)
ax.set_ylabel("Probes (top 50 by |log2FC|)", fontsize=10)
for i, c in enumerate(col_colors):
    ax.add_patch(plt.Rectangle((i, -0.5), 1, 0.5, color=c,
                                clip_on=False, transform=ax.get_xaxis_transform()))
plt.tight_layout()
fig.savefig(FIG_DIR / "02d_heatmap_top50.png", dpi=300)
plt.close()
print("  Saved -> 02d_heatmap_top50.png")

print("Generating p-value distribution plot...")
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(deg_stats["p_value"], bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
ax.set_xlabel("Raw p-value", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of Raw p-values (Welch's t-test)", fontsize=13)
ax.axvline(x=0.05, color="red", linestyle="--", linewidth=1, label="p = 0.05")
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(FIG_DIR / "02e_pvalue_distribution.png", dpi=300)
plt.close()
print("  Saved -> 02e_pvalue_distribution.png")

print("\n" + "=" * 55)
print("DEG ANALYSIS SUMMARY")
print("=" * 55)
print(f"  Total probes tested   : {len(deg_stats)}")
print(f"  Significant DEGs      : {len(degs)}")
print(f"    Upregulated         : {len(up_degs)}")
print(f"    Downregulated       : {len(down_degs)}")
print(f"  Threshold             : adj.p < {P_THRESH}, |log2FC| >= {FC_THRESH}")
print(f"  Method                : Welch's t-test + BH FDR correction")
print("=" * 55)
print("Done. Proceed to 03_feature_selection.py")
