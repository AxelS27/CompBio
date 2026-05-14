# Research Context: Leakage-Controlled ML for Cholangiocarcinoma Biomarker Identification

## Current Study Design

This project now uses **GSE76297** as the primary paired-patient cohort for CCA tumor versus non-tumor classification and **GSE26566** as an external validation cohort.

- Primary cohort: GSE76297, Affymetrix HTA 2.0
- External cohort: GSE26566, Illumina HumanRef-8 v2.0
- Primary task: classify CCA tumor tissue versus matched non-tumor tissue
- Validation principle: patient-level grouping for paired samples

## Why The Pipeline Was Corrected

The previous internal cross-validation used a global `deg_significant.csv` file as the candidate feature pool. Because that DEG table was computed using all samples before cross-validation, validation folds indirectly influenced feature selection. This produced overly optimistic internal metrics.

The corrected pipeline now repeats supervised DEG prefiltering and feature selection **inside each training fold** during model evaluation. Validation folds are used only for scoring.

## Corrected Pipeline

1. `src/01_preprocessing.py`
   - Parse GSE76297.
   - Keep CCA tumor and CCA non-tumor samples.
   - Log2 check, IQR filtering, quantile normalization.
   - Save `expr_processed.csv` and `sample_metadata.csv`.

2. `src/02_deg_analysis.py`
   - Run full-cohort DEG analysis for biological interpretation and visualization.
   - This output is not used as a fixed global feature pool for corrected CV.

3. `src/04_classification.py`
   - Run 5-fold `StratifiedGroupKFold` by patient.
   - Inside each fold:
     - compute training-only DEG prefilter,
     - run LASSO, SVM-RFE, or RF feature selection on training data only,
     - fit scaler and SMOTE on training data only,
     - evaluate on held-out patients.
   - Save corrected `classification_results.csv`.
   - Save final full-cohort feature sets for biomarker discovery only.

4. `src/05_biomarker.py`
   - Train final SVM-RFE + Logistic Regression biomarker model on the full primary cohort.
   - Save `biomarkers_ranked.csv` and enrichment outputs.

5. `src/07_external_validation.py`
   - Map GSE76297 and GSE26566 probes to common gene symbols using compact probe-map caches in `output/results`.
   - Train on GSE76297 and test on GSE26566.
   - Save `external_validation_results.csv`.

6. `src/06_final_figures.py`
   - Generate exactly five final figures:
     - preprocessing QC,
     - comparative CV AUC heatmap,
     - external validation AUC,
     - ROC curves,
     - overfitting/underfitting learning curve.
   - ROC and learning-curve plots repeat feature selection inside the relevant training split/fold.

7. `src/07_generate_summary.py`
   - Generate `output/research_summary.docx` from the current result files.

## Main Methodological Notes

- Patient-level grouping is required because most GSE76297 samples are tumor/non-tumor pairs from the same patient.
- Full-cohort DEG plots are useful for biological interpretation but should not be used to claim unbiased CV performance.
- Corrected internal CV metrics remain very high after leakage control, so they should be described as strong internal performance rather than "perfect" proof.
- External validation on GSE26566 remains the most important generalization check because it tests cross-cohort and cross-platform robustness.
- The `dataset/` directory intentionally stores only the two GSE expression matrices. Platform annotation is represented by compact probe-map CSV files in `output/results/`.
