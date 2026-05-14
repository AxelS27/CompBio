import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, matthews_corrcoef)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
RES_DIR  = ROOT / "output" / "results"

print("=" * 55)
print("Step 1: Checking required dataset and compact map files")
print("=" * 55)

GSE26566_FILE = DATA_DIR / "GSE26566_series_matrix.txt"
GPL6104_MAP   = RES_DIR / "probe_map_gpl6104.csv"
GPL17586_MAP  = RES_DIR / "probe_map_gpl17586.csv"

missing = [p.name for p in [GSE26566_FILE, GPL6104_MAP, GPL17586_MAP] if not p.exists()]
if missing:
    print(f"\nERROR: missing required files: {', '.join(missing)}")
    print("Dataset folder should contain only GSE files; compact probe maps live in output/results.")
    raise SystemExit(1)

print("\n" + "=" * 55)
print("Step 2: Loading compact probe maps")
print("=" * 55)

def load_probe_map(path):
    df = pd.read_csv(path)
    mapping = dict(zip(df["probe_id"].astype(str), df["gene_symbol"].astype(str)))
    print(f"  {path.name}: {len(mapping):,} probes -> gene symbols")
    return mapping

affy_probe2gene = load_probe_map(GPL17586_MAP)
illum_probe2gene = load_probe_map(GPL6104_MAP)

print("\n" + "=" * 55)
print("Step 3: Loading GSE26566 (external cohort)")
print("=" * 55)

sample_titles26 = []
sample_gsm26    = []
sample_tissues26 = []

with open(GSE26566_FILE, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("!Sample_title"):
            sample_titles26 = [p.strip('"') for p in line.split("\t")[1:]]
        elif line.startswith("!Sample_geo_accession"):
            sample_gsm26 = [p.strip('"') for p in line.split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1") and "tissue:" in line:
            sample_tissues26 = [p.strip('"').replace("tissue: ", "")
                                 for p in line.split("\t")[1:]]

print(f"  Samples in header: {len(sample_gsm26)}")

with open(GSE26566_FILE, "r", encoding="utf-8", errors="replace") as f:
    expr26 = pd.read_csv(f, sep="\t", comment="!", index_col=0, low_memory=False)
expr26 = expr26[~expr26.index.astype(str).str.startswith("!")]
expr26 = expr26.apply(pd.to_numeric, errors="coerce")
expr26.dropna(how="all", inplace=True)
print(f"  Raw matrix: {expr26.shape}")

meta26 = pd.DataFrame({
    "gsm":    sample_gsm26,
    "title":  sample_titles26,
    "tissue": sample_tissues26 if sample_tissues26 else [""] * len(sample_gsm26),
})

def assign_label26(row):
    t = row.get("tissue", "")
    title = row.get("title", "")
    if "Cholangiocarcinoma" in t or "CCA" in title:
        return 1
    if "Surrounding liver" in t or "Normal intrahepatic" in t or "normal" in t.lower():
        return 0
    return None

meta26["label"] = meta26.apply(assign_label26, axis=1)
meta26 = meta26.dropna(subset=["label"]).copy()
meta26["label"] = meta26["label"].astype(int)

common_gsm26 = [g for g in meta26["gsm"].values if g in expr26.columns]
expr26 = expr26[common_gsm26]
meta26 = meta26[meta26["gsm"].isin(common_gsm26)].set_index("gsm")
print(f"  CCA samples: {len(meta26)}  "
      f"(tumor={meta26['label'].sum()}, non-tumor={(meta26['label']==0).sum()})")

# Log2 transform if needed
if expr26.values.max() > 100:
    mn = expr26.values.min()
    if mn <= 0:
        expr26 = expr26 + abs(mn) + 1
    expr26 = np.log2(expr26)
    print("  Applied log2 transform")
else:
    print("  Already log2-scaled")

print("\n" + "=" * 55)
print("Step 4: Mapping probe IDs -> gene-level expression")
print("=" * 55)

expr76 = pd.read_csv(RES_DIR / "expr_processed.csv", index_col=0)
meta76 = pd.read_csv(RES_DIR / "sample_metadata.csv", index_col=0)

def probes_to_genes(expr_df, probe2gene):
    expr_df = expr_df.copy()
    expr_df["gene"] = expr_df.index.map(lambda p: probe2gene.get(str(p), ""))
    expr_df = expr_df[expr_df["gene"] != ""]
    expr_df = expr_df[~expr_df["gene"].str.startswith("---")]
    # Mean expression when multiple probes map to same gene
    gene_expr = expr_df.groupby("gene").mean()
    return gene_expr

print("  Mapping GSE76297 (Affymetrix HTA 2.0)...")
gene76 = probes_to_genes(expr76, affy_probe2gene)
print(f"    {expr76.shape[0]:,} probes -> {gene76.shape[0]:,} unique genes")

print("  Mapping GSE26566 (Illumina HumanRef-8)...")
gene26 = probes_to_genes(expr26, illum_probe2gene)
print(f"    {expr26.shape[0]:,} probes -> {gene26.shape[0]:,} unique genes")

common_genes = sorted(set(gene76.index) & set(gene26.index))
print(f"\n  Common genes between both platforms: {len(common_genes):,}")

if len(common_genes) < 50:
    print("  ERROR: Too few common genes. Check annotation parsing.")
    raise SystemExit(1)

gene76_common = gene76.loc[common_genes, meta76.index].T   # samples × genes
gene26_common = gene26.loc[common_genes, meta26.index].T   # samples × genes

y76 = meta76["label"].values
y26 = meta26["label"].values

print(f"  GSE76297 (train) : {gene76_common.shape}  "
      f"(tumor={y76.sum()}, non-tumor={(y76==0).sum()})")
print(f"  GSE26566 (test)  : {gene26_common.shape}  "
      f"(tumor={y26.sum()}, non-tumor={(y26==0).sum()})")

print("\n" + "=" * 55)
print("Step 5: Cross-dataset classification")
print("=" * 55)

X_train = gene76_common.values
X_test  = gene26_common.values
gene_names = np.array(common_genes)

print("  Feature selection (LASSO) on GSE76297 train set ...")
sc_fs   = StandardScaler()
X_tr_sc = sc_fs.fit_transform(X_train)

lcv = LogisticRegressionCV(
    Cs=np.logspace(-3, 1, 20), cv=5, penalty="l1",
    solver="liblinear", max_iter=2000, random_state=42, scoring="roc_auc"
)
lcv.fit(X_tr_sc, y76)
coef = np.abs(lcv.coef_[0])
nz   = np.where(coef > 0)[0]
if len(nz) < 5:
    nz = np.argsort(coef)[-50:]
sel_genes = gene_names[nz]
print(f"  LASSO selected {len(sel_genes)} genes: {list(sel_genes[:8])}...")

X_tr = gene76_common[sel_genes].values
X_te = gene26_common[sel_genes].values

scaler    = StandardScaler()
X_tr_sc2  = scaler.fit_transform(X_tr)
X_te_sc   = scaler.transform(X_te)

k = max(1, min(5, np.bincount(y76).min() - 1))
smote = SMOTE(random_state=42, k_neighbors=k)
X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_sc2, y76)

n_neg = (y76 == 0).sum()
n_pos = (y76 == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=2000, random_state=42, class_weight="balanced"),
    "SVM": SVC(
        kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
    "LightGBM": LGBMClassifier(
        n_estimators=300, random_state=42, class_weight="balanced",
        verbose=-1, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, random_state=42, scale_pos_weight=n_neg/n_pos,
        eval_metric="logloss", verbosity=0, n_jobs=-1),
    "Naive Bayes": GaussianNB(),
}

ext_results = []
print(f"\n  {'Model':<22} {'AUC':>6} {'Acc':>6} {'F1':>6} {'MCC':>6}")
print("  " + "-" * 48)

for name, model in models.items():
    model.fit(X_tr_sm, y_tr_sm)
    y_pred = model.predict(X_te_sc)
    y_prob = model.predict_proba(X_te_sc)[:, 1]

    auc = round(roc_auc_score(y26, y_prob), 4)
    acc = round(accuracy_score(y26, y_pred), 4)
    f1  = round(f1_score(y26, y_pred, average="macro"), 4)
    mcc = round(matthews_corrcoef(y26, y_pred), 4)
    print(f"  {name:<22} {auc:>6.3f} {acc:>6.3f} {f1:>6.3f} {mcc:>6.3f}")
    ext_results.append({"Model": name, "AUC": auc, "Accuracy": acc,
                        "F1": f1, "MCC": mcc})

ext_df = pd.DataFrame(ext_results).sort_values("AUC", ascending=False)
ext_df.to_csv(RES_DIR / "external_validation_results.csv", index=False)
print(f"\n  Saved -> external_validation_results.csv")

print("\n" + "=" * 55)
print("EXTERNAL VALIDATION SUMMARY")
print("=" * 55)
print(f"  Train : GSE76297 (n={len(y76)}, {n_pos} tumor, {n_neg} non-tumor)")
print(f"  Test  : GSE26566 (n={len(y26)}, {y26.sum()} tumor, {(y26==0).sum()} non-tumor)")
print(f"  Common genes used: {len(sel_genes)}")
best = ext_df.iloc[0]
print(f"  Best model : {best['Model']}")
print(f"  Best AUC   : {best['AUC']:.4f}")
print(f"  Best F1    : {best['F1']:.4f}")
print(f"  Best MCC   : {best['MCC']:.4f}")
print("=" * 55)
print("Done. Run 06_final_figures.py to update figures.")
