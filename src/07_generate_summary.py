import pandas as pd
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "output" / "results"
FIG_DIR = ROOT / "output" / "figures"
OUT_DIR = ROOT / "output"

expr = pd.read_csv(RES_DIR / "expr_processed.csv", index_col=0)
meta = pd.read_csv(RES_DIR / "sample_metadata.csv", index_col=0)
degs = pd.read_csv(RES_DIR / "deg_significant.csv", index_col=0)
results = pd.read_csv(RES_DIR / "classification_results.csv")
bio_df = pd.read_csv(RES_DIR / "biomarkers_ranked.csv")
ext_path = RES_DIR / "external_validation_results.csv"
ext_results = pd.read_csv(ext_path) if ext_path.exists() else pd.DataFrame()

best = results.sort_values(["AUC", "MCC"], ascending=False).iloc[0]
best_ext = ext_results.sort_values("AUC", ascending=False).iloc[0] if not ext_results.empty else None

tumor_n = int((meta["label"] == 1).sum())
non_tumor_n = int((meta["label"] == 0).sum())
patient_ids = meta["title"].str.extract(r"(LCS_\d+)", expand=False)

doc = Document()
section = doc.sections[0]
section.top_margin = Inches(0.7)
section.bottom_margin = Inches(0.7)
section.left_margin = Inches(0.75)
section.right_margin = Inches(0.75)

styles = doc.styles
styles["Normal"].font.name = "Arial"
styles["Normal"].font.size = Pt(10)

title = doc.add_heading("Comparative Machine Learning for Cholangiocarcinoma Biomarker Identification", 0)
title.alignment = 1
subtitle = doc.add_paragraph(
    "Leakage-controlled patient-level validation on GSE76297 with external validation on GSE26566"
)
subtitle.alignment = 1

doc.add_heading("Executive Summary", level=1)
doc.add_paragraph(
    "This updated pipeline evaluates CCA tumor versus non-tumor classification using GSE76297 as "
    "the primary paired-patient cohort. The main methodological correction is that supervised DEG "
    "prefiltering and feature selection are repeated inside each training fold during cross-validation. "
    "Validation folds are therefore not used to choose candidate probes."
)
doc.add_paragraph(
    f"The primary cohort contains {len(meta)} samples from {patient_ids.nunique()} patients "
    f"({tumor_n} tumor, {non_tumor_n} non-tumor). The final full-cohort DEG analysis identified "
    f"{len(degs)} significant probes for downstream biological interpretation."
)

doc.add_heading("Corrected Validation Design", level=1)
items = [
    "Patient-level grouping keeps paired tumor/non-tumor samples from the same patient in the same fold.",
    "Within each outer CV fold, DEG prefiltering is fit only on the training samples.",
    "LASSO, SVM-RFE, and Random Forest feature selection are fit only after that training-only prefilter.",
    "Standard scaling and SMOTE are fit only on training folds.",
    "Final biomarker ranking is trained on the full primary cohort for discovery, not for reporting CV performance.",
    "GSE26566 is retained as an independent external validation cohort across platforms.",
]
for item in items:
    doc.add_paragraph(item, style="List Bullet")

doc.add_heading("Primary Cross-Validation Results", level=1)
doc.add_paragraph(
    f"Best corrected internal combination: {best['Pipeline']} + {best['Model']} "
    f"(AUC {best['AUC']:.3f} +/- {best['AUC_std']:.3f}, "
    f"Accuracy {best['Accuracy']:.3f}, F1 {best['F1']:.3f}, MCC {best['MCC']:.3f})."
)

tbl = doc.add_table(rows=1, cols=7)
tbl.style = "Table Grid"
headers = ["Pipeline", "Model", "AUC", "Accuracy", "F1", "Sensitivity", "MCC"]
for cell, header in zip(tbl.rows[0].cells, headers):
    cell.text = header

for _, row in results.sort_values(["AUC", "MCC"], ascending=False).head(10).iterrows():
    cells = tbl.add_row().cells
    values = [
        row["Pipeline"],
        row["Model"],
        f"{row['AUC']:.3f}",
        f"{row['Accuracy']:.3f}",
        f"{row['F1']:.3f}",
        f"{row['Sensitivity']:.3f}",
        f"{row['MCC']:.3f}",
    ]
    for cell, value in zip(cells, values):
        cell.text = str(value)

if best_ext is not None:
    doc.add_heading("External Validation", level=1)
    doc.add_paragraph(
        "The external validation step trains on GSE76297 gene-level expression and tests on GSE26566 "
        "after mapping both platforms to common gene symbols."
    )
    doc.add_paragraph(
        f"Best external model: {best_ext['Model']} "
        f"(AUC {best_ext['AUC']:.3f}, Accuracy {best_ext['Accuracy']:.3f}, "
        f"F1 {best_ext['F1']:.3f}, MCC {best_ext['MCC']:.3f})."
    )
    tbl_ext = doc.add_table(rows=1, cols=5)
    tbl_ext.style = "Table Grid"
    for cell, header in zip(tbl_ext.rows[0].cells, ["Model", "AUC", "Accuracy", "F1", "MCC"]):
        cell.text = header
    for _, row in ext_results.sort_values("AUC", ascending=False).iterrows():
        cells = tbl_ext.add_row().cells
        for cell, value in zip(
            cells,
            [row["Model"], f"{row['AUC']:.3f}", f"{row['Accuracy']:.3f}", f"{row['F1']:.3f}", f"{row['MCC']:.3f}"],
        ):
            cell.text = str(value)

doc.add_heading("Candidate Biomarkers", level=1)
doc.add_paragraph(
    "Candidate biomarkers are ranked from the final full-cohort SVM-RFE + Logistic Regression model. "
    "These rankings are intended for biological interpretation and follow-up validation."
)
tbl_bio = doc.add_table(rows=1, cols=4)
tbl_bio.style = "Table Grid"
for cell, header in zip(tbl_bio.rows[0].cells, ["Rank", "Probe ID", "Gene Symbol", "Direction"]):
    cell.text = header
for rank, (_, row) in enumerate(bio_df.head(20).iterrows(), start=1):
    cells = tbl_bio.add_row().cells
    values = [rank, row["probe_id"], row["gene_symbol"], row["direction"]]
    for cell, value in zip(cells, values):
        cell.text = str(value)

doc.add_heading("Generated Outputs", level=1)
outputs = [
    ("classification_results.csv", "Corrected nested patient-level CV results."),
    ("external_validation_results.csv", "Cross-dataset validation on GSE26566."),
    ("biomarkers_ranked.csv", "Final candidate biomarker ranking."),
    ("fig1_preprocessing_qc.png", "Preprocessing QC: normalized expression and PCA separation."),
    ("fig2_cv_auc_heatmap.png", "Comparative model AUC heatmap across feature-selection pipelines."),
    ("fig3_external_validation.png", "External validation AUC on GSE26566."),
    ("fig4_roc_svmrfe.png", "ROC curves with train-split-only SVM-RFE feature selection."),
    ("fig5_overfitting_diagnostic.png", "Learning curve with nested feature selection inside each fold."),
]
for name, desc in outputs:
    exists = "available" if (RES_DIR / name).exists() or (FIG_DIR / name).exists() else "missing"
    doc.add_paragraph(f"{name}: {desc} ({exists}).", style="List Bullet")

doc.add_heading("Verified Supporting Literature", level=1)
doc.add_paragraph(
    "The following references were checked against publisher or repository pages and are used only for "
    "claims they directly support."
)

refs = [
    (
        "Zhu et al., 2025",
        "Scientific Reports",
        "LightGBM diagnostic model for CCA using APOF/DIO1/OTC; supports CCA ML benchmarking context.",
        "https://doi.org/10.1038/s41598-025-30431-5",
    ),
    (
        "Negrini et al., 2020",
        "PMC / metabolomic CCA screening",
        "Compared six ML models for CCA screening; Naive Bayes and SVM reached high AUC on holdout data.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC7460348/",
    ),
    (
        "Shen et al., 2022",
        "Frontiers in Oncology",
        "Gene-chip meta-analysis for CCA biomarkers using GEO datasets including GSE26566.",
        "https://doi.org/10.3389/fonc.2022.1001400",
    ),
    (
        "Fan et al., 2024",
        "Nature Communications",
        "Transcriptome-based CCA molecular classification and CORE-37 prognostic biomarker.",
        "https://doi.org/10.1038/s41467-024-44748-8",
    ),
    (
        "Liu et al., 2021",
        "Scientific Reports",
        "Large-cohort CCA transcriptomic landscape using GSE76297 and GSE26566 among other cohorts.",
        "https://doi.org/10.1038/s41598-021-93250-4",
    ),
    (
        "Alharbi & Vakanski, 2023",
        "Bioengineering",
        "Review of machine learning methods for cancer classification using gene expression data.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC9952758/",
    ),
    (
        "Alhenawi et al., 2022",
        "Computers in Biology and Medicine",
        "Systematic review of feature selection methods for microarray cancer classification.",
        "https://doi.org/10.1016/j.compbiomed.2021.105051",
    ),
    (
        "Zanella et al., 2022",
        "International Journal of Molecular Sciences",
        "Comparative study of feature selectors and classifiers for high-dimensional gene-expression classification.",
        "https://doi.org/10.3390/ijms23169087",
    ),
]

tbl_ref = doc.add_table(rows=1, cols=4)
tbl_ref.style = "Table Grid"
for cell, header in zip(tbl_ref.rows[0].cells, ["Reference", "Source", "Relevance", "Link / DOI"]):
    cell.text = header
for ref in refs:
    cells = tbl_ref.add_row().cells
    for cell, value in zip(cells, ref):
        cell.text = value

doc.add_heading("Interpretation Note", level=1)
doc.add_paragraph(
    "The corrected internal CV results are very high and the nested learning curve does not indicate "
    "overfitting or underfitting. These results should still be described as strong internal performance "
    "rather than proof of perfection; external validation on GSE26566 is used as the main generalization check."
)

out_path = OUT_DIR / "research_summary.docx"
doc.save(out_path)
print(f"Saved updated summary -> {out_path}")
