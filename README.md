# Pfizer - HCP Segmentation & Propensity Modeling in Ulcerative Colitis

This repository contains the end-to-end machine learning pipelines, notebooks, and final models designed for Healthcare Professional (HCP) segmentation and propensity scoring in the context of Pfizer's Ulcerative Colitis treatments.

---

## Repository Structure

The project is structured into the following directories on the `main` branch:

*   **[HistGradientBoosting/](file:///Users/davidbazalduamendez/Documents/GitHub/Pfizer-segmentation-Ulcerative-Colitis/HistGradientBoosting)**: Contains the binary classifier designed to separate SEG_A HCPs from SEG_B and SEG_C HCPs.
*   **[Catboost clean/](file:///Users/davidbazalduamendez/Documents/GitHub/Pfizer-segmentation-Ulcerative-Colitis/Catboost%20clean)**: Implements the propensity scoring pipeline, probability calibration, and explainability analytics.
*   **[3-class XGBoost/](file:///Users/davidbazalduamendez/Documents/GitHub/Pfizer-segmentation-Ulcerative-Colitis/3-class%20XGBoost)**: Houses the multiclass classifier focused on segmenting HCPs into three distinct classes (SEG_A, SEG_B, and SEG_C).
*   **[models/](file:///Users/davidbazalduamendez/Documents/GitHub/Pfizer-segmentation-Ulcerative-Colitis/models)**: Auxiliary configurations and intermediate data files shared across pipelines.

---

## Final Models Saved in `main`

Three final models are saved and maintained in this branch, each addressing a specific business objective:

### 1. Hist Gradient Boosting Classifier (Binary Classification)
*   **Objective:** Distinguish **SEG_A** HCPs from **SEG_B** and **SEG_C** HCPs (SEG_BC).
*   **Model Artifact:** [`best_binary_segA_vs_segBC.joblib`](file:///Users/davidbazalduamendez/Documents/GitHub/Pfizer-segmentation-Ulcerative-Colitis/HistGradientBoosting/Model%20artifacts/best_binary_segA_vs_segBC.joblib) (accompanied by metadata in [`model_metadata.json`](file:///Users/davidbazalduamendez/Documents/GitHub/Pfizer-segmentation-Ulcerative-Colitis/HistGradientBoosting/Model%20artifacts/model_metadata.json)).
*   **Evaluation Metrics (Test Set):**
    | Metric | Value |
    | :--- | :---: |
    | **Accuracy** | 77.23% |
    | **Macro F1-Score** | 77.04% |
    | **ROC-AUC** | 85.18% |
    | **SEG_A Recall** | 80.17% |
    | **SEG_BC Recall** | 73.79% |

### 2. CatBoost Clean (Propensity Scoring & Calibration)
*   **Objective:** Predict the calibrated probability (propensity score) of an HCP prescribing Pfizer's brand, specifically focusing on evaluating performance in high-potential B/C segments.
*   **Key Pipeline Features:**
    *   **Isotonic Calibration**: Adjusts the raw classifier outputs to output true, reliable probability scores.
    *   **SHAP Glass-Box Explainability**: Extracts global feature importance and computes the **Top 5 local reason codes** (prescriptive drivers) for each HCP.
*   **Outputs & Visualizations:**
    *   Final predictions and reason codes are exported to [`propensity_predictions_with_reasons.parquet`](file:///Users/davidbazalduamendez/Documents/GitHub/Pfizer-segmentation-Ulcerative-Colitis/Catboost%20clean/output/propensity_predictions_with_reasons.parquet).
    *   Calibration performance diagram is saved in [`calibration_analysis.png`](file:///Users/davidbazalduamendez/Documents/GitHub/Pfizer-segmentation-Ulcerative-Colitis/Catboost%20clean/output/calibration_analysis.png) and the global SHAP beeswarm plot is saved in [`shap_global_summary.png`](file:///Users/davidbazalduamendez/Documents/GitHub/Pfizer-segmentation-Ulcerative-Colitis/Catboost%20clean/output/shap_global_summary.png).

### 3. XGBoost Classifier (3-Class HCP Segmentation)
*   **Objective:** Categorize HCPs directly into the three target classes (SEG_A, SEG_B, and SEG_C).
*   **Model Artifact:** [`best_xgb_multiclass_atseg_candidate58.joblib`](file:///Users/davidbazalduamendez/Documents/GitHub/Pfizer-segmentation-Ulcerative-Colitis/3-class%20XGBoost/Model%20artifacts/best_xgb_multiclass_atseg_candidate58.joblib) (detailed in [`xgb_multiclass_candidate58_metadata.json`](file:///Users/davidbazalduamendez/Documents/GitHub/Pfizer-segmentation-Ulcerative-Colitis/3-class%20XGBoost/Model%20artifacts/xgb_multiclass_candidate58_metadata.json)).
*   **Selection Note:** Candidate 58 was selected for deployment due to its superior recall on **SEG_C** while maintaining a highly competitive macro F1-score.
*   **Evaluation Metrics (Test Set):**
    | Metric | Value |
    | :--- | :---: |
    | **Accuracy** | 64.50% |
    | **Macro F1-Score** | 57.93% |
    | **Weighted F1-Score** | 64.80% |
    | **SEG_A Recall** | 76.27% |
    | **SEG_B Recall** | 57.01% |
    | **SEG_C Recall** | 41.03% |

---

## xperimental Models & Historical Runs

> [!NOTE]
> To inspect other evaluated models, hyperparameter tuning runs, and historical experiments, please switch to **other repository branches** (particularly **`dev-bazaldua`**). The `main` branch is strictly reserved for production-grade pipelines and final serialized model artifacts.