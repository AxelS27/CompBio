import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, matthews_corrcoef)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from scipy import stats
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

ROOT    = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "output" / "results"

print("Loading data...")
expr_norm = pd.read_csv(RES_DIR / "expr_processed.csv", index_col=0)
meta      = pd.read_csv(RES_DIR / "sample_metadata.csv", index_col=0)

y           = meta["label"].values
all_probes  = expr_norm.index.astype(str).values
X_all       = expr_norm.loc[all_probes, meta.index].T.values
# Each LCS_XXX patient has a paired tumor (A) and non-tumor (B) sample; group by patient
# to prevent the model from learning patient-specific profiles instead of tumor biology.
patient_ids = meta["title"].str.extract(r"(LCS_\d+)", expand=False).values

print(f"  Samples           : {len(y)}  ({np.bincount(y)[0]} non-tumor, {np.bincount(y)[1]} tumor)")
print(f"  Unique patients   : {len(set(patient_ids))}")
print(f"  Probe pool        : {len(all_probes)}")

N_TARGET = 50
MAX_DEG_POOL = 500

def select_deg_pool(X_tr, y_tr, probe_ids, max_features=MAX_DEG_POOL):
    """Training-fold-only supervised prefilter to avoid validation leakage."""
    tumor = X_tr[y_tr == 1]
    non_tumor = X_tr[y_tr == 0]
    t_stats, p_vals = stats.ttest_ind(tumor, non_tumor, axis=0, equal_var=False)
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

def select_lasso(X_tr, y_tr, probe_ids, n=N_TARGET):
    sc  = StandardScaler()
    Xsc = sc.fit_transform(X_tr)
    lcv = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 20), cv=5, penalty="l1",
        solver="liblinear", max_iter=2000, random_state=42, scoring="roc_auc",
    )
    lcv.fit(Xsc, y_tr)
    coef  = np.abs(lcv.coef_[0])
    nz    = np.where(coef > 0)[0]
    if len(nz) == 0:
        nz = np.argsort(coef)[-n:]
    order = np.argsort(coef[nz])[::-1][:n]
    return probe_ids[nz[order]]

def select_svmrfe(X_tr, y_tr, probe_ids, n=N_TARGET):
    sc  = StandardScaler()
    Xsc = sc.fit_transform(X_tr)
    rfe = RFE(LinearSVC(C=0.1, max_iter=5000, random_state=42),
              n_features_to_select=min(n, X_tr.shape[1]), step=50)
    rfe.fit(Xsc, y_tr)
    return probe_ids[rfe.support_]

def select_rf(X_tr, y_tr, probe_ids, n=N_TARGET):
    rf = RandomForestClassifier(n_estimators=300, random_state=42,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X_tr, y_tr)
    order = np.argsort(rf.feature_importances_)[::-1][:n]
    return probe_ids[order]

def make_models(y_train):
    n_neg = np.bincount(y_train)[0]
    n_pos = np.bincount(y_train)[1]
    return {
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

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sens = spec = 0
    return {
        "AUC":         round(roc_auc_score(y_true, y_prob), 4),
        "Accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "F1":          round(f1_score(y_true, y_pred, average="macro"), 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "MCC":         round(matthews_corrcoef(y_true, y_pred), 4),
    }

print("\n5-fold patient-level cross-validation...")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

fs_selectors = {"LASSO": select_lasso, "SVM-RFE": select_svmrfe, "RF": select_rf}

model_names = list(make_models(y).keys())
fold_results       = {(fs, m): [] for fs in fs_selectors for m in model_names}
train_fold_results = {(fs, m): [] for fs in fs_selectors for m in model_names}

for fold_i, (idx_train, idx_val) in enumerate(
        cv.split(X_all, y, groups=patient_ids)):
    print(f"\n  Fold {fold_i+1}/5  "
          f"(train={len(idx_train)}, val={len(idx_val)}, "
          f"val patients: {len(set(patient_ids[idx_val]))})")

    assert len(set(patient_ids[idx_train]) & set(patient_ids[idx_val])) == 0

    X_tr_all, X_val_all = X_all[idx_train], X_all[idx_val]
    y_tr, y_val         = y[idx_train], y[idx_val]
    fold_probe_pool     = select_deg_pool(X_tr_all, y_tr, all_probes)
    X_tr_pool           = expr_norm.loc[fold_probe_pool, meta.index].T.values[idx_train]
    print(f"    Training-only DEG pool: {len(fold_probe_pool)} probes")

    for fs_name, selector in fs_selectors.items():
        probe_list = selector(X_tr_pool, y_tr, fold_probe_pool)
        valid      = [p for p in probe_list if p in expr_norm.index]
        X_fs       = expr_norm.loc[valid, meta.index].T.values
        X_tr_raw   = X_fs[idx_train]
        X_val_raw  = X_fs[idx_val]

        scaler     = StandardScaler()
        X_tr_sc    = scaler.fit_transform(X_tr_raw)
        X_val_sc   = scaler.transform(X_val_raw)

        k = max(1, min(5, np.bincount(y_tr).min() - 1))
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_sc, y_tr)

        for model_name, model in make_models(y_tr).items():
            model.fit(X_tr_sm, y_tr_sm)
            y_pred_val = model.predict(X_val_sc)
            y_prob_val = model.predict_proba(X_val_sc)[:, 1]
            m = compute_metrics(y_val, y_pred_val, y_prob_val)
            fold_results[(fs_name, model_name)].append(m)

            # Evaluate train AUC on original (non-augmented) training data
            y_prob_tr = model.predict_proba(X_tr_sc)[:, 1]
            train_fold_results[(fs_name, model_name)].append(
                round(roc_auc_score(y_tr, y_prob_tr), 4))

            if fold_i == 0:  # print fold 1 only
                print(f"    [{fs_name} | {model_name}] "
                      f"AUC={m['AUC']:.3f}  Acc={m['Accuracy']:.3f}  "
                      f"F1={m['F1']:.3f}  MCC={m['MCC']:.3f}")

print("\n\nAggregating CV results...")
all_results = []
for fs_name in fs_selectors:
    for model_name in model_names:
        folds   = fold_results[(fs_name, model_name)]
        tr_aucs = train_fold_results[(fs_name, model_name)]
        row = {"Pipeline": fs_name, "Model": model_name}
        for metric in ["AUC","Accuracy","F1","Sensitivity","Specificity","MCC"]:
            vals = [f[metric] for f in folds]
            row[metric]          = round(np.mean(vals), 4)
            row[f"{metric}_std"] = round(np.std(vals),  4)
        row["Train_AUC"]     = round(np.mean(tr_aucs), 4)
        row["Train_AUC_std"] = round(np.std(tr_aucs),  4)
        all_results.append(row)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(["AUC","MCC"], ascending=False).reset_index(drop=True)

cols = ["Pipeline","Model","AUC","AUC_std","Accuracy","Accuracy_std",
        "F1","F1_std","Sensitivity","Specificity","MCC","MCC_std",
        "Train_AUC","Train_AUC_std"]
results_df = results_df[cols]
results_df.to_csv(RES_DIR / "classification_results.csv", index=False)

print("\nFULL CV RESULTS TABLE (mean +/- std across 5 folds):")
display = results_df[["Pipeline","Model","AUC","AUC_std","Accuracy","F1","MCC","Train_AUC"]].copy()
display["AUC"] = display.apply(lambda r: f"{r['AUC']:.3f}+/-{r['AUC_std']:.3f}", axis=1)
print(display[["Pipeline","Model","AUC","Accuracy","F1","MCC","Train_AUC"]].to_string(index=False))

best_row = results_df.iloc[0]
print(f"\nBest combination : [{best_row['Pipeline']}] + [{best_row['Model']}]")
print(f"  CV AUC   : {best_row['AUC']:.4f} +/- {best_row['AUC_std']:.4f}")
print(f"  CV F1    : {best_row['F1']:.4f} +/- {best_row['F1_std']:.4f}")
print(f"  CV MCC   : {best_row['MCC']:.4f} +/- {best_row['MCC_std']:.4f}")
print(f"  Train AUC: {best_row['Train_AUC']:.4f} +/- {best_row['Train_AUC_std']:.4f}")

print("\nSaving best feature set (fit on all data for biomarker analysis)...")
final_probe_pool = select_deg_pool(X_all, y, all_probes)
X_final_pool     = expr_norm.loc[final_probe_pool, meta.index].T.values
probes_lasso     = select_lasso(X_final_pool,  y, final_probe_pool)
probes_svmrfe    = select_svmrfe(X_final_pool, y, final_probe_pool)
probes_rf        = select_rf(X_final_pool,     y, final_probe_pool)
pd.DataFrame({"probe_id": probes_lasso}).to_csv(RES_DIR / "features_lasso.csv",   index=False)
pd.DataFrame({"probe_id": probes_svmrfe}).to_csv(RES_DIR / "features_svmrfe.csv", index=False)
pd.DataFrame({"probe_id": probes_rf}).to_csv(RES_DIR / "features_rf.csv",         index=False)
print(f"  Final full-cohort DEG pool: {len(final_probe_pool)} probes")
print(f"  LASSO: {len(probes_lasso)}  SVM-RFE: {len(probes_svmrfe)}  RF: {len(probes_rf)}")

print("\n" + "=" * 55)
print("CLASSIFICATION SUMMARY (5-fold patient CV)")
print("=" * 55)
print(f"  Best : {best_row['Pipeline']} + {best_row['Model']}")
print(f"  AUC  : {best_row['AUC']:.4f} +/- {best_row['AUC_std']:.4f}")
print(f"  F1   : {best_row['F1']:.4f} +/- {best_row['F1_std']:.4f}")
print(f"  MCC  : {best_row['MCC']:.4f} +/- {best_row['MCC_std']:.4f}")
print("=" * 55)
print("Done. Proceed to 05_biomarker.py")
