import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "dataset" / "GSE76297_series_matrix.txt"
FIG_DIR   = ROOT / "output" / "figures"
RES_DIR   = ROOT / "output" / "results"

FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

print("Parsing series matrix header...")

sample_titles = []
sample_gsm    = []

with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("!Sample_title"):
            sample_titles = [p.strip('"') for p in line.split("\t")[1:]]
        elif line.startswith("!Sample_geo_accession"):
            sample_gsm = [p.strip('"') for p in line.split("\t")[1:]]

print(f"  Found {len(sample_gsm)} samples in metadata")

print("Reading expression table (this may take a moment)...")

expr_df = pd.read_csv(DATA_FILE, sep="\t", comment="!", index_col=0, low_memory=False)
expr_df = expr_df[~expr_df.index.astype(str).str.startswith("!")]
expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
expr_df.dropna(how="all", inplace=True)

print(f"  Raw shape: {expr_df.shape}  (probes x samples)")

meta = pd.DataFrame({"gsm": sample_gsm, "title": sample_titles})

def assign_label(title):
    if "CCA Tumor" in title:
        return 1
    elif "CCA Non-Tumor" in title:
        return 0
    return None  # HCC and other samples excluded

meta["label"] = meta["title"].apply(assign_label)
meta = meta.dropna(subset=["label"]).copy()
meta["label"] = meta["label"].astype(int)

print("\nFinal label distribution (CCA only):")
print(meta["label"].value_counts().rename({0: "Non-tumor (0)", 1: "CCA tumor (1)"}).to_string())

common_gsm = [g for g in meta["gsm"].values if g in expr_df.columns]
expr_df    = expr_df[common_gsm].copy()
meta       = meta[meta["gsm"].isin(common_gsm)].set_index("gsm")

print(f"\nAligned expression matrix: {expr_df.shape}")

print("\nChecking if log2 transform is needed...")
max_val  = float(np.nanmax(expr_df.values))
mean_val = float(np.nanmean(expr_df.values))
print(f"  Max value : {max_val:.2f}")
print(f"  Mean value: {mean_val:.2f}")

if max_val > 100:
    print("  -> Raw/linear scale detected. Shifting to positive then applying log2.")
    min_val = float(np.nanmin(expr_df.values))
    if min_val <= 0:
        shift = abs(min_val) + 1
        print(f"  -> Min value is {min_val:.2f}, shifting by {shift:.2f} before log2.")
        expr_df = expr_df + shift
    expr_df = np.log2(expr_df)
else:
    print("  -> Already log2-transformed. Skipping.")

expr_df.replace([np.inf, -np.inf], np.nan, inplace=True)
nan_count = int(expr_df.isna().sum().sum())
if nan_count > 0:
    print(f"  Imputing {nan_count} NaN values with row medians...")
    expr_df = expr_df.apply(lambda row: row.fillna(row.median()), axis=1)

print(f"  After transform -- max: {expr_df.values.max():.2f}, mean: {expr_df.values.mean():.2f}")

print(f"\nProbe count before IQR filtering: {expr_df.shape[0]}")

iqr_vals = expr_df.apply(
    lambda row: float(np.percentile(row.dropna(), 75) - np.percentile(row.dropna(), 25)),
    axis=1
)

print(f"  IQR stats -- min: {iqr_vals.min():.4f}, mean: {iqr_vals.mean():.4f}, max: {iqr_vals.max():.4f}")

threshold     = float(np.percentile(iqr_vals, 20))
expr_filtered = expr_df[iqr_vals > threshold]

print(f"Probes after IQR filtering (threshold IQR > {threshold:.4f}): {expr_filtered.shape[0]}")

print("\nApplying quantile normalization across samples...")

arr    = expr_filtered.values.copy()
sort_i = np.argsort(arr, axis=0)
rank_m = np.sort(arr, axis=0).mean(axis=1)

for j in range(arr.shape[1]):
    arr[sort_i[:, j], j] = rank_m

expr_norm = pd.DataFrame(arr, index=expr_filtered.index, columns=expr_filtered.columns)
print(f"  Done. Shape: {expr_norm.shape}")

expr_norm.to_csv(RES_DIR / "expr_processed.csv")
meta.to_csv(RES_DIR / "sample_metadata.csv")

print(f"\nSaved:")
print(f"  Expression matrix -> {RES_DIR / 'expr_processed.csv'}")
print(f"  Sample metadata   -> {RES_DIR / 'sample_metadata.csv'}")

print("\nGenerating QC plots...")

tumor_cols    = meta[meta["label"] == 1].index.tolist()
nontumor_cols = meta[meta["label"] == 0].index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

np.random.seed(42)
pick = np.random.choice(expr_norm.columns, min(30, expr_norm.shape[1]), replace=False)
expr_norm[pick].boxplot(ax=axes[0], vert=False, showfliers=False)
axes[0].set_title("Sample Expression Distribution (30 random)", fontsize=10)
axes[0].set_xlabel("Expression (log2)")
axes[0].set_yticks([])

tumor_means    = expr_norm[tumor_cols].mean(axis=1)
nontumor_means = expr_norm[nontumor_cols].mean(axis=1)

axes[1].hist(tumor_means, bins=80, alpha=0.6, color="firebrick",
             label=f"CCA Tumor (n={len(tumor_cols)})", density=True)
axes[1].hist(nontumor_means, bins=80, alpha=0.6, color="steelblue",
             label=f"Non-tumor (n={len(nontumor_cols)})", density=True)
axes[1].set_title("Mean Expression Distribution by Group", fontsize=10)
axes[1].set_xlabel("Mean expression (log2)")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
fig.savefig(FIG_DIR / "01_preprocessing_qc.png", dpi=150)
plt.close()
print(f"  QC plot saved -> {FIG_DIR / '01_preprocessing_qc.png'}")

print("\n" + "=" * 55)
print("PREPROCESSING SUMMARY")
print("=" * 55)
print(f"  Dataset              : GSE76297 (Affymetrix HTA 2.0)")
print(f"  Total CCA samples    : {expr_norm.shape[1]}")
print(f"  CCA tumor  (label=1) : {len(tumor_cols)}")
print(f"  Non-tumor  (label=0) : {len(nontumor_cols)}")
print(f"  Probes retained      : {expr_norm.shape[0]}")
print(f"  Class ratio          : {len(tumor_cols)}:{len(nontumor_cols)} "
      f"({len(tumor_cols)/len(nontumor_cols):.1f}:1)")
print("=" * 55)
print("Done. Proceed to 02_deg_analysis.py")
