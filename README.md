# Churn Prediction Model With Interactive Evaluation Suite

This repository contains a complete binary churn prediction pipeline built with TensorFlow and scikit-learn, plus interactive Plotly dashboards, SHAP explainability, K-fold validation, threshold simulation, and a unified artifacts report.

## Highlights

- End-to-end training and evaluation pipeline for customer churn (`0` = not churned, `1` = churned)
- Neural network with regularization, class balancing, and threshold optimization
- Interactive Plotly dashboards for model diagnostics
- K-fold cross-validation summary and charts
- SHAP feature importance export
- Dash threshold simulator app
- Unified landing report linking all generated artifacts

## Latest Model Snapshot

- Baseline Accuracy: `0.9317`
- Baseline ROC AUC: `0.9701`
- Baseline PR AUC: `0.9654`
- Best Threshold (F1-optimized): `0.8511`
- Tuned Accuracy: `0.9500`
- Tuned F1: `0.9143`

Source: `metrics_summary.json`

## Repository Structure

- `churn prediction.py` - Main training/evaluation script
- `threshold_dashboard_app.py` - Interactive threshold tuning app (Dash)
- `model_artifacts_report.html` - Unified report entry point
- `model_dashboard_plotly.html` - Main training/evaluation dashboard
- `kfold_dashboard_plotly.html` - K-fold performance dashboard
- `shap_feature_importance_plotly.html` - SHAP importance dashboard
- `metrics_summary.json` - Key metrics snapshot
- `training_history.csv` - Epoch-level training log
- `threshold_sweep.csv` - Threshold-by-metric sweep data
- `kfold_metrics.csv` - Per-fold validation metrics
- `shap_feature_importance.csv` - Feature importance scores
- `test_predictions.csv` - Test set predictions
- `best_model.keras` - Best model checkpoint

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the full pipeline:

```bash
python "churn prediction.py"
```

4. Build unified report only (no training):

```bash
python "churn prediction.py" --build-report-only
```

5. Run threshold simulator app:

```bash
python "churn prediction.py" --run-threshold-app
```

## Open Artifacts

Start with `model_artifacts_report.html` and use:

- Open All Dashboards
- Individual dashboard launch buttons
- Artifact links and status table

## Notes

- If `customer_data.csv` is not found, the script generates a synthetic dataset for demonstration.
- On native Windows with TensorFlow >= 2.11, training typically runs on CPU unless using WSL2 or DirectML setup.

## GitHub Push Syntax

From the project folder, run:

```bash
cd "C:/Users/Ebi_Mahmdli/Desktop/churn-prediction-model"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If remote `origin` already exists, use:

```bash
git remote set-url origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
