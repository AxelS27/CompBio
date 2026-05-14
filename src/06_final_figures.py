import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "output" / "results"
FIG_DIR = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FINAL_FIGURES = {
    "fig1_preprocessing_qc.png",
    "fig2_cv_auc_heatmap.png",
    "fig3_external_validation.png",
    "fig4_roc_svmrfe.png",
    "fig5_overfitting_diagnostic.png",
}

for path in FIG_DIR.glob("*.png"):
    if path.name not in FINAL_FIGURES:
        path.unlink()

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

expr_norm = pd.read_csv(RES_DIR / "expr_processed.csv", index_col=0)
meta = pd.read_csv(RES_DIR / "sample_metadata.csv", index_col=0)
results = pd.read_csv(RES_DIR / "classification_results.csv")
ext_results = pd.read_csv(RES_DIR / "external_validation_results.csv")

UP_COLOR = "#C0392B"
DOWN_COLOR = "#2980B9"
ACCENT = "#27AE60"
WARN = "#E67E22"
N_TARGET = 50
MAX_DEG_POOL = 500


def select_deg_pool(X_tr, y_tr, probe_ids, max_features=MAX_DEG_POOL):
    tumor = X_tr[y_tr == 1]
    non_tumor = X_tr[y_tr == 0]
    _, p_vals = stats.ttest_ind(tumor, non_tumor, axis=0, equal_var=False)
    p_vals = np.nan_to_num(p_vals, nan=1.0, posinf=1.0, neginf=1.0)
    _, adj_p, _, _ = multipletests(p_vals, method="fdr_bh")
    log2fc = tumor.mean(axis=0) - non_tumor.mean(axis=0)
    sig = np.where((adj_p < 0.05) & (np.abs(log2fc) >= 1.0))[0]
    if len(sig) == 0:
        order = np.lexsort((-np.abs(log2fc), adj_p))
        sig = order[:max_features]
    elif len(sig) > max_features:
        sig = sig[np.argsort(adj_p[sig])[:max_features]]
    return probe_ids[sig]


def select_svmrfe(X_tr, y_tr, probe_ids, n=N_TARGET):
    scaler = StandardScaler()
    Xsc = scaler.fit_transform(X_tr)
    rfe = RFE(
        LinearSVC(C=0.1, max_iter=5000, random_state=42),
        n_features_to_select=min(n, X_tr.shape[1]),
        step=50,
    )
    rfe.fit(Xsc, y_tr)
    return probe_ids[rfe.support_]


def model_dict(y_train):
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    return {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=300, random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            random_state=42,
            scale_pos_weight=n_neg / n_pos,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        ),
        "Naive Bayes": GaussianNB(),
    }


print("Generating Figure 1 - Preprocessing QC...")
y = meta["label"].values
labels = np.where(y == 1, "Tumor", "Non-tumor")
sample_means = expr_norm.mean(axis=0)
sample_iqr = expr_norm.quantile(0.75, axis=0) - expr_norm.quantile(0.25, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(x=labels, y=sample_means.values, ax=axes[0], palette=[DOWN_COLOR, UP_COLOR])
axes[0].set_title("Sample Mean Expression After Normalization")
axes[0].set_xlabel("")
axes[0].set_ylabel("Mean log2 expression")

pca_input = StandardScaler().fit_transform(expr_norm.T.values)
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(pca_input)
pca_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "Class": labels})
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Class", palette={"Tumor": UP_COLOR, "Non-tumor": DOWN_COLOR}, s=45, ax=axes[1])
axes[1].set_title("PCA of Processed Expression Matrix")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
axes[1].legend(frameon=False)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_preprocessing_qc.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Generating Figure 2 - Comparative CV AUC heatmap...")
pivot = results.pivot(index="Pipeline", columns="Model", values="AUC")
fig, ax = plt.subplots(figsize=(9.5, 4.5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.90, vmax=1.00, linewidths=0.5, cbar_kws={"label": "AUC"}, ax=ax)
ax.set_title("Comparative Model Performance - Patient-Level CV AUC")
ax.set_xlabel("Model")
ax.set_ylabel("Feature Selection Pipeline")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig2_cv_auc_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Generating Figure 3 - External validation...")
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ext_plot = ext_results.sort_values("AUC", ascending=True)
colors = [ACCENT if model == ext_results.sort_values("AUC", ascending=False).iloc[0]["Model"] else "#7F8C8D" for model in ext_plot["Model"]]
ax.barh(ext_plot["Model"], ext_plot["AUC"], color=colors, alpha=0.9)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Random")
ax.set_xlim(0.45, 1.0)
ax.set_xlabel("External Validation AUC on GSE26566")
ax.set_title("Cross-Cohort Generalization: Train GSE76297, Test GSE26566")
for i, auc in enumerate(ext_plot["AUC"]):
    ax.text(auc + 0.01, i, f"{auc:.3f}", va="center", fontsize=9)
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_external_validation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Generating Figure 4 - ROC curves...")
probe_ids = expr_norm.index.astype(str).values
X_all = expr_norm.loc[probe_ids, meta.index].T.values
patient_ids = meta["title"].str.extract(r"(LCS_\d+)", expand=False).values
idx_train, idx_test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X_all, y, groups=patient_ids))
roc_pool = select_deg_pool(X_all[idx_train], y[idx_train], probe_ids)
roc_selected = select_svmrfe(expr_norm.loc[roc_pool, meta.index].T.values[idx_train], y[idx_train], roc_pool)
X = expr_norm.loc[roc_selected, meta.index].T.values
scaler = StandardScaler()
X_tr = scaler.fit_transform(X[idx_train])
X_te = scaler.transform(X[idx_test])
k = max(1, min(5, np.bincount(y[idx_train]).min() - 1))
X_sm, y_sm = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y[idx_train])

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random (AUC=0.500)")
palette = ["#C0392B", "#2980B9", "#27AE60", "#8E44AD", "#E67E22", "#16A085"]
for (name, model), color in zip(model_dict(y[idx_train]).items(), palette):
    model.fit(X_sm, y_sm)
    y_prob = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y[idx_test], y_prob)
    auc = roc_auc_score(y[idx_test], y_prob)
    ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"{name} (AUC={auc:.3f})")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - SVM-RFE Pipeline")
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig4_roc_svmrfe.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Generating Figure 5 - Overfitting/underfitting diagnostic...")
unique_pts = np.unique(patient_ids)
np.random.seed(42)
np.random.shuffle(unique_pts)
train_fracs = [0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
train_mean, train_std, val_mean, val_std = [], [], [], []
base_model = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")

for frac in train_fracs:
    n_pts = max(10, int(len(unique_pts) * frac))
    selected_pts = set(unique_pts[:n_pts])
    selected_idx = np.where([pid in selected_pts for pid in patient_ids])[0]
    X_sub = X_all[selected_idx]
    y_sub = y[selected_idx]
    p_sub = patient_ids[selected_idx]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr_scores, va_scores = [], []

    for idx_tr, idx_va in cv.split(X_sub, y_sub, groups=p_sub):
        Xtr_all = X_sub[idx_tr]
        Xva_all = X_sub[idx_va]
        ytr = y_sub[idx_tr]
        yva = y_sub[idx_va]
        pool = select_deg_pool(Xtr_all, ytr, probe_ids)
        tr_df = pd.DataFrame(Xtr_all, columns=probe_ids)
        va_df = pd.DataFrame(Xva_all, columns=probe_ids)
        selected = select_svmrfe(tr_df[pool].values, ytr, pool)
        Xtr = tr_df[selected].values
        Xva = va_df[selected].values
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xva = sc.transform(Xva)
        k = max(1, min(5, np.bincount(ytr).min() - 1))
        Xsm, ysm = SMOTE(random_state=42, k_neighbors=k).fit_resample(Xtr, ytr)
        base_model.fit(Xsm, ysm)
        tr_scores.append(roc_auc_score(ytr, base_model.predict_proba(Xtr)[:, 1]))
        va_scores.append(roc_auc_score(yva, base_model.predict_proba(Xva)[:, 1]))

    train_mean.append(np.mean(tr_scores))
    train_std.append(np.std(tr_scores))
    val_mean.append(np.mean(va_scores))
    val_std.append(np.std(va_scores))

sizes = [int(len(unique_pts) * f) * 2 for f in train_fracs]
tm, ts = np.array(train_mean), np.array(train_std)
vm, vs = np.array(val_mean), np.array(val_std)
gap = tm[-1] - vm[-1]
gap_color = WARN if gap > 0.05 else ACCENT

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(sizes, tm, "o-", color=UP_COLOR, linewidth=2, label="Training AUC")
ax.fill_between(sizes, tm - ts, tm + ts, alpha=0.15, color=UP_COLOR)
ax.plot(sizes, vm, "s--", color=DOWN_COLOR, linewidth=2, label="Validation AUC")
ax.fill_between(sizes, vm - vs, vm + vs, alpha=0.15, color=DOWN_COLOR)
ax.annotate(
    f"Final gap = {gap:.3f}",
    xy=(sizes[-1], (tm[-1] + vm[-1]) / 2),
    xytext=(sizes[-2], max(0.55, (tm[-1] + vm[-1]) / 2 - 0.08)),
    fontsize=9,
    color=gap_color,
    fontweight="bold",
    arrowprops=dict(arrowstyle="-", color=gap_color, lw=1.2),
)
ax.set_xlabel("Approx. Training Samples")
ax.set_ylabel("AUC")
ax.set_ylim(0.5, 1.08)
ax.set_title("Overfitting/Underfitting Diagnostic - Nested Learning Curve")
ax.text(
    0.02,
    0.04,
    "Underfitting: both curves low. Overfitting: large train-validation gap.\n"
    "This plot repeats DEG filtering and SVM-RFE inside each fold.",
    transform=ax.transAxes,
    fontsize=8,
    color="dimgray",
    va="bottom",
)
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig5_overfitting_diagnostic.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nFinal figure set:")
for name in sorted(FINAL_FIGURES):
    print(f"  {name}")
