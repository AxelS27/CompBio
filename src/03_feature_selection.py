import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

ROOT    = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "output" / "results"
FIG_DIR = ROOT / "output" / "figures"

print("Loading preprocessed data...")

expr_norm = pd.read_csv(RES_DIR / "expr_processed.csv", index_col=0)
meta      = pd.read_csv(RES_DIR / "sample_metadata.csv", index_col=0)
degs      = pd.read_csv(RES_DIR / "deg_significant.csv", index_col=0)

print(f"  Expression matrix : {expr_norm.shape}")
print(f"  Significant DEGs  : {len(degs)}")

common_probes = [p for p in degs.index if p in expr_norm.index]
X = expr_norm.loc[common_probes, meta.index].T.values   # samples x probes
y = meta["label"].values
probe_ids = np.array(common_probes)

print(f"  X shape: {X.shape}  (samples x DEG probes)")
print(f"  Class distribution: {np.bincount(y)}  [0=non-tumor, 1=tumor]")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Target ~50-100 genes for downstream ML (manageable, interpretable)
N_TARGET = 50

print("\n[Pipeline A] LASSO feature selection...")

from sklearn.linear_model import LogisticRegressionCV

lasso_cv = LogisticRegressionCV(
    Cs=np.logspace(-3, 1, 20),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    penalty="l1",
    solver="liblinear",
    max_iter=2000,
    random_state=42,
    scoring="roc_auc",
)
lasso_cv.fit(X_scaled, y)

lasso_coef   = np.abs(lasso_cv.coef_[0])
lasso_nonzero = np.where(lasso_coef > 0)[0]
lasso_genes   = probe_ids[lasso_nonzero]
lasso_scores  = lasso_coef[lasso_nonzero]

order_A       = np.argsort(lasso_scores)[::-1][:N_TARGET]
genes_A       = lasso_genes[order_A]
scores_A      = lasso_scores[order_A]

print(f"  Best C           : {lasso_cv.C_[0]:.5f}")
print(f"  Non-zero coefs   : {len(lasso_nonzero)}")
print(f"  Selected (top {N_TARGET}): {len(genes_A)}")

print("\n[Pipeline B] SVM-RFE feature selection...")

svm_rfe = RFE(
    estimator=LinearSVC(C=0.1, max_iter=5000, random_state=42),
    n_features_to_select=N_TARGET,
    step=50,
)
svm_rfe.fit(X_scaled, y)

rfe_mask   = svm_rfe.support_
genes_B    = probe_ids[rfe_mask]
ranks_B    = svm_rfe.ranking_[rfe_mask]
scores_B   = 1.0 / ranks_B   # rank 1 = best

order_B    = np.argsort(scores_B)[::-1]
genes_B    = genes_B[order_B]
scores_B   = scores_B[order_B]

print(f"  Selected features: {len(genes_B)}")

print("\n[Pipeline C] Random Forest importance...")

rf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
rf.fit(X_scaled, y)

rf_importance = rf.feature_importances_
order_C   = np.argsort(rf_importance)[::-1][:N_TARGET]
genes_C   = probe_ids[order_C]
scores_C  = rf_importance[order_C]

print(f"  Selected (top {N_TARGET}): {len(genes_C)}")

pd.DataFrame({"probe_id": genes_A, "importance": scores_A}).to_csv(
    RES_DIR / "features_lasso.csv", index=False)
pd.DataFrame({"probe_id": genes_B, "importance": scores_B}).to_csv(
    RES_DIR / "features_svmrfe.csv", index=False)
pd.DataFrame({"probe_id": genes_C, "importance": scores_C}).to_csv(
    RES_DIR / "features_rf.csv", index=False)

print(f"\nGene subsets saved -> {RES_DIR}")

set_A = set(genes_A)
set_B = set(genes_B)
set_C = set(genes_C)

overlap_AB  = set_A & set_B
overlap_AC  = set_A & set_C
overlap_BC  = set_B & set_C
overlap_ABC = set_A & set_B & set_C

print(f"\nOverlap Analysis:")
print(f"  LASSO  n={len(set_A)}, SVM-RFE n={len(set_B)}, RF n={len(set_C)}")
print(f"  LASSO  & SVM-RFE   : {len(overlap_AB)}")
print(f"  LASSO  & RF        : {len(overlap_AC)}")
print(f"  SVM-RFE & RF       : {len(overlap_BC)}")
print(f"  All 3 methods      : {len(overlap_ABC)}")
if overlap_ABC:
    print(f"  Consensus probes   : {', '.join(sorted(str(g) for g in overlap_ABC))}")

pd.DataFrame({"probe_id": sorted(overlap_ABC)}).to_csv(
    RES_DIR / "features_consensus.csv", index=False)

print("\nGenerating Venn diagram...")

try:
    from matplotlib_venn import venn3
    fig, ax = plt.subplots(figsize=(7, 6))
    venn3([set_A, set_B, set_C],
          set_labels=("LASSO", "SVM-RFE", "Random Forest"),
          set_colors=("#e74c3c", "#3498db", "#2ecc71"),
          alpha=0.6, ax=ax)
    ax.set_title("Feature Overlap Across Selection Methods", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03a_venn_overlap.png", dpi=300)
    plt.close()
    print("  Saved -> 03a_venn_overlap.png")
except ImportError:
    print("  matplotlib-venn not installed, generating bar chart instead...")
    fig, ax = plt.subplots(figsize=(6, 4))
    cats   = ["LASSO\nonly", "SVM-RFE\nonly", "RF\nonly",
              "LASSO+\nSVM-RFE", "LASSO\n+RF", "SVM-RFE\n+RF", "All 3"]
    counts = [
        len(set_A - set_B - set_C),
        len(set_B - set_A - set_C),
        len(set_C - set_A - set_B),
        len(overlap_AB - set_C),
        len(overlap_AC - set_B),
        len(overlap_BC - set_A),
        len(overlap_ABC),
    ]
    colors = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#e67e22","#1abc9c","#f39c12"]
    ax.bar(cats, counts, color=colors, edgecolor="white")
    ax.set_ylabel("Number of probes")
    ax.set_title("Feature Overlap Across Selection Methods", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03a_venn_overlap.png", dpi=300)
    plt.close()
    print("  Saved -> 03a_venn_overlap.png (bar chart)")

print("Generating LASSO importance plot...")

fig, ax = plt.subplots(figsize=(8, 9))
top_n = min(30, len(genes_A))
ax.barh(range(top_n), scores_A[:top_n][::-1],
        color="#e74c3c", edgecolor="none", alpha=0.85)
ax.set_yticks(range(top_n))
ax.set_yticklabels([str(g) for g in genes_A[:top_n][::-1]], fontsize=8)
ax.set_xlabel("|LASSO Coefficient|", fontsize=11)
ax.set_title(f"LASSO: Top {top_n} Selected Features", fontsize=12)
plt.tight_layout()
fig.savefig(FIG_DIR / "03b_lasso_importance.png", dpi=300)
plt.close()
print("  Saved -> 03b_lasso_importance.png")

print("Generating SVM-RFE importance plot...")

fig, ax = plt.subplots(figsize=(8, 9))
top_n = min(30, len(genes_B))
ax.barh(range(top_n), scores_B[:top_n][::-1],
        color="#3498db", edgecolor="none", alpha=0.85)
ax.set_yticks(range(top_n))
ax.set_yticklabels([str(g) for g in genes_B[:top_n][::-1]], fontsize=8)
ax.set_xlabel("Inverse Rank Score", fontsize=11)
ax.set_title(f"SVM-RFE: Top {top_n} Selected Features", fontsize=12)
plt.tight_layout()
fig.savefig(FIG_DIR / "03c_svmrfe_importance.png", dpi=300)
plt.close()
print("  Saved -> 03c_svmrfe_importance.png")

print("Generating RF importance plot...")

fig, ax = plt.subplots(figsize=(8, 9))
top_n = min(30, len(genes_C))
ax.barh(range(top_n), scores_C[:top_n][::-1],
        color="#2ecc71", edgecolor="none", alpha=0.85)
ax.set_yticks(range(top_n))
ax.set_yticklabels([str(g) for g in genes_C[:top_n][::-1]], fontsize=8)
ax.set_xlabel("Gini Importance", fontsize=11)
ax.set_title(f"Random Forest: Top {top_n} Features by Importance", fontsize=12)
plt.tight_layout()
fig.savefig(FIG_DIR / "03d_rf_importance.png", dpi=300)
plt.close()
print("  Saved -> 03d_rf_importance.png")

print("Generating pipeline summary plot...")

fig, ax = plt.subplots(figsize=(7, 5))
methods = ["LASSO", "SVM-RFE", "Random Forest", "Consensus\n(all 3)"]
n_genes = [len(set_A), len(set_B), len(set_C), len(overlap_ABC)]
colors  = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
bars    = ax.bar(methods, n_genes, color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, n_genes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Selected Features", fontsize=11)
ax.set_title("Features Selected per Method", fontsize=13)
ax.set_ylim(0, max(n_genes) * 1.15)
plt.tight_layout()
fig.savefig(FIG_DIR / "03e_feature_count_summary.png", dpi=300)
plt.close()
print("  Saved -> 03e_feature_count_summary.png")

print("\n" + "=" * 55)
print("FEATURE SELECTION SUMMARY")
print("=" * 55)
print(f"  Input DEG probes   : {len(common_probes)}")
print(f"  Target N per method: {N_TARGET}")
print(f"  LASSO selected     : {len(set_A)}")
print(f"  SVM-RFE selected   : {len(set_B)}")
print(f"  RF selected        : {len(set_C)}")
print(f"  Consensus (all 3)  : {len(overlap_ABC)}")
print("=" * 55)
print("Done. Proceed to 04_classification.py")
