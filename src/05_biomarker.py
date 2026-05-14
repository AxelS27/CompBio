import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
warnings.filterwarnings("ignore")
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import gseapy as gp

ROOT    = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "output" / "results"
FIG_DIR = ROOT / "output" / "figures"
DATA_DIR = ROOT / "dataset"

SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{1,24}$")

def clean_symbol(symbol):
    symbol = str(symbol).strip()
    if (
        symbol
        and symbol not in ("---", "NULL")
        and SYMBOL_RE.match(symbol)
        and "=" not in symbol
        and ".hg." not in symbol
    ):
        return symbol
    return ""

def is_mapped_gene(symbol):
    return clean_symbol(symbol) == str(symbol).strip()

print("Loading compact GPL17586 probe map...")
MAP_FILE = RES_DIR / "probe_map_gpl17586.csv"
probe_to_gene = {}
if MAP_FILE.exists():
    map_df = pd.read_csv(MAP_FILE)
    probe_to_gene = dict(zip(map_df["probe_id"].astype(str), map_df["gene_symbol"].astype(str)))
    print(f"  Loaded {len(probe_to_gene)} probe-to-gene mappings")
else:
    print("  Compact probe map not found, using probe IDs as fallback gene names")

def probe_to_symbol(probe_id):
    s = probe_to_gene.get(str(probe_id), str(probe_id))
    return clean_symbol(s) or str(probe_id)

print("\nLoading data & retraining best model (SVM-RFE + Logistic Regression)...")

expr_norm  = pd.read_csv(RES_DIR / "expr_processed.csv", index_col=0)
meta       = pd.read_csv(RES_DIR / "sample_metadata.csv", index_col=0)
svmrfe_df  = pd.read_csv(RES_DIR / "features_svmrfe.csv")
svmrfe_df["probe_id"] = svmrfe_df["probe_id"].astype(str)

probes     = [p for p in svmrfe_df["probe_id"].tolist() if p in expr_norm.index]
gene_names = [probe_to_symbol(p) for p in probes]

y   = meta["label"].values
X   = expr_norm.loc[probes, meta.index].T.values

scaler       = StandardScaler()
X_sc         = scaler.fit_transform(X)

smote        = SMOTE(random_state=42, k_neighbors=min(5, np.bincount(y).min()-1))
X_train_sm, y_train_sm = smote.fit_resample(X_sc, y)

model = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
model.fit(X_train_sm, y_train_sm)

coef_abs    = np.abs(model.coef_[0])
coef_signed = model.coef_[0]

biomarker_df = pd.DataFrame({
    "probe_id":     probes,
    "gene_symbol":  gene_names,
    "coefficient":  coef_signed,
    "abs_coef":     coef_abs,
    "direction":    ["Up in Tumor" if c > 0 else "Down in Tumor" for c in coef_signed],
})
biomarker_df = biomarker_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
biomarker_df.to_csv(RES_DIR / "biomarkers_ranked.csv", index=False)

print(f"  Total features     : {len(biomarker_df)}")
print(f"  Up in tumor        : {(biomarker_df['direction']=='Up in Tumor').sum()}")
print(f"  Down in tumor      : {(biomarker_df['direction']=='Down in Tumor').sum()}")
print("\n  Top 20 Biomarkers:")
print(biomarker_df[["gene_symbol","coefficient","direction"]].head(20).to_string(index=False))

top_genes = biomarker_df["gene_symbol"].head(30).tolist()
top_genes_clean = [g for g in top_genes if is_mapped_gene(g) and not g.endswith("_st")]

print("\nGenerating biomarker coefficient plot...")

top30 = biomarker_df.head(30).sort_values("coefficient")
colors = ["#2980b9" if v < 0 else "#c0392b" for v in top30["coefficient"]]

fig, ax = plt.subplots(figsize=(9, 10))
ax.barh(range(len(top30)), top30["coefficient"], color=colors, edgecolor="none", alpha=0.85)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30["gene_symbol"], fontsize=9)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Logistic Regression Coefficient", fontsize=12)
ax.set_title("Top 30 Biomarker Genes\n(SVM-RFE + Logistic Regression)", fontsize=12)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#c0392b", label="Up in Tumor"),
                   Patch(color="#2980b9", label="Down in Tumor")], fontsize=10)
plt.tight_layout()
fig.savefig(FIG_DIR / "05a_biomarker_coefficients.png", dpi=300)
plt.close()
print("  Saved -> 05a_biomarker_coefficients.png")

print("Generating expression boxplot of top 10 genes...")

tumor_cols    = meta[meta["label"] == 1].index.tolist()
nontumor_cols = meta[meta["label"] == 0].index.tolist()
top10_probes  = biomarker_df["probe_id"].head(10).tolist()
top10_genes   = biomarker_df["gene_symbol"].head(10).tolist()

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
axes = axes.flatten()

for i, (probe, gene) in enumerate(zip(top10_probes, top10_genes)):
    ax = axes[i]
    t_vals  = expr_norm.loc[probe, tumor_cols].values
    nt_vals = expr_norm.loc[probe, nontumor_cols].values
    ax.boxplot([nt_vals, t_vals], patch_artist=True,
               medianprops=dict(color="black", linewidth=1.5),
               boxprops=dict(facecolor="none"),
               widths=0.5)
    ax.set_facecolor("#f9f9f9")
    parts = ax.boxplot([nt_vals, t_vals], patch_artist=True,
                       medianprops=dict(color="black", linewidth=1.5),
                       widths=0.5)
    parts["boxes"][0].set_facecolor("#aed6f1")
    parts["boxes"][1].set_facecolor("#f1948a")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Non-tumor", "Tumor"], fontsize=8)
    ax.set_title(gene, fontsize=10, fontweight="bold")
    ax.set_ylabel("log2 Expr", fontsize=7)

plt.suptitle("Expression Distribution of Top 10 Biomarker Genes", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "05b_top10_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved -> 05b_top10_boxplot.png")

print("\nRunning pathway enrichment analysis via Enrichr...")

gene_list = top_genes_clean if len(top_genes_clean) >= 5 else [
    g for g in biomarker_df["gene_symbol"].tolist() if is_mapped_gene(g)]

print(f"  Submitting {len(gene_list)} gene symbols: {gene_list[:10]}...")

enrichment_results = {}
libraries = {
    "GO_BP":   "GO_Biological_Process_2023",
    "GO_MF":   "GO_Molecular_Function_2023",
    "KEGG":    "KEGG_2021_Human",
}

for key, lib in libraries.items():
    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=lib,
            organism="human",
            outdir=None,
            verbose=False,
        )
        df = enr.results
        df = df[df["Adjusted P-value"] < 0.05].copy() if not df.empty else df
        df = df.sort_values("Adjusted P-value").head(20)
        enrichment_results[key] = df
        print(f"  {key}: {len(df)} significant terms (adj.p < 0.05)")
    except Exception as e:
        print(f"  {key}: failed ({e})")
        enrichment_results[key] = pd.DataFrame()

for key, df in enrichment_results.items():
    if not df.empty:
        df.to_csv(RES_DIR / f"enrichment_{key}.csv", index=False)

def plot_enrichment(df, title, fname, color):
    if df.empty:
        print(f"  No significant terms for {title}, skipping plot.")
        return
    df = df.head(15).copy()
    df["-log10(adj.p)"] = -np.log10(df["Adjusted P-value"].clip(lower=1e-10))
    df["Gene_count"] = df["Overlap"].apply(
        lambda x: int(x.split("/")[0]) if "/" in str(x) else 1)
    df = df.sort_values("-log10(adj.p)")

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(df["-log10(adj.p)"], range(len(df)),
                    s=df["Gene_count"] * 20, c=df["-log10(adj.p)"],
                    cmap=color, alpha=0.85, edgecolors="gray", linewidths=0.4)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([t[:65] for t in df["Term"]], fontsize=8)
    ax.set_xlabel("-log10(Adjusted P-value)", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.colorbar(sc, ax=ax, label="-log10(adj.p)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=300)
    plt.close()
    print(f"  Saved -> {fname}")

print("\nGenerating enrichment dotplots...")
plot_enrichment(enrichment_results.get("GO_BP", pd.DataFrame()),
                "GO Biological Process Enrichment\n(Top 15 Terms)",
                "05c_go_bp_dotplot.png", "Reds")

plot_enrichment(enrichment_results.get("GO_MF", pd.DataFrame()),
                "GO Molecular Function Enrichment\n(Top 15 Terms)",
                "05d_go_mf_dotplot.png", "Oranges")

plot_enrichment(enrichment_results.get("KEGG", pd.DataFrame()),
                "KEGG Pathway Enrichment\n(Top 15 Terms)",
                "05e_kegg_dotplot.png", "Blues")

print("Generating combined enrichment barplot...")

combined_rows = []
for key, label in [("GO_BP","GO-BP"), ("GO_MF","GO-MF"), ("KEGG","KEGG")]:
    df = enrichment_results.get(key, pd.DataFrame())
    if not df.empty:
        for _, row in df.head(10).iterrows():
            combined_rows.append({
                "Term":    row["Term"][:55],
                "neglog10p": -np.log10(max(row["Adjusted P-value"], 1e-10)),
                "Database": label,
            })

if combined_rows:
    comb_df = pd.DataFrame(combined_rows).sort_values("neglog10p", ascending=True)
    pal = {"GO-BP": "#e74c3c", "GO-MF": "#f39c12", "KEGG": "#3498db"}

    fig, ax = plt.subplots(figsize=(10, max(6, len(comb_df)*0.35)))
    for i, (_, row) in enumerate(comb_df.iterrows()):
        ax.barh(i, row["neglog10p"], color=pal[row["Database"]], alpha=0.85, edgecolor="none")
    ax.set_yticks(range(len(comb_df)))
    ax.set_yticklabels(comb_df["Term"], fontsize=8)
    ax.set_xlabel("-log10(Adjusted P-value)", fontsize=11)
    ax.set_title("Pathway & GO Enrichment of Top Biomarker Genes", fontsize=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for l, c in pal.items()], fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05f_enrichment_combined_bar.png", dpi=300)
    plt.close()
    print("  Saved -> 05f_enrichment_combined_bar.png")
else:
    print("  No enrichment results to plot.")

print("\n" + "=" * 55)
print("BIOMARKER SUMMARY")
print("=" * 55)
print(f"  Best model         : SVM-RFE + Logistic Regression")
print(f"  Total features     : {len(biomarker_df)}")
print(f"  Genes mapped       : {sum(is_mapped_gene(g) for g in biomarker_df['gene_symbol'])}")
print(f"  Up in tumor        : {(biomarker_df['direction']=='Up in Tumor').sum()}")
print(f"  Down in tumor      : {(biomarker_df['direction']=='Down in Tumor').sum()}")
for key, df in enrichment_results.items():
    print(f"  {key} sig terms   : {len(df)}")
print("\n  Top 10 Candidate Biomarkers:")
for _, row in biomarker_df.head(10).iterrows():
    print(f"    {row['gene_symbol']:15s}  coef={row['coefficient']:+.4f}  {row['direction']}")
print("=" * 55)
print("Done. All pipeline steps complete.")
