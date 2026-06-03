# Churn Risk Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3D9970)
![Dash](https://img.shields.io/badge/Dash-App-1F77B4)
![Status](https://img.shields.io/badge/Model-Ready-success)

End-to-end customer churn prediction system with neural network training, threshold optimization, explainability, and interactive dashboards.

## What This Project Does

- Predicts customer churn probability from tabular features.
- Optimizes decision threshold for stronger precision/recall tradeoff.
- Evaluates with ROC, PR, confusion matrix, and threshold sweeps.
- Runs K-fold validation and exports aggregate fold analytics.
- Produces SHAP feature importance for model explainability.
- Generates a unified artifact report page for one-click navigation.

## Latest Snapshot

- Baseline Accuracy: 0.9317
- Baseline ROC AUC: 0.9701
- Baseline PR AUC: 0.9654
- Best Threshold (F1-optimized): 0.8511
- Tuned Accuracy: 0.9500
- Tuned F1: 0.9143

Source: metrics_summary.json

## Dashboard Previews

Main evaluation dashboard:

![Main Dashboard](assets/main-dashboard.png)

K-fold cross-validation dashboard:

![KFold Dashboard](assets/kfold-dashboard.png)

SHAP feature importance dashboard:

![SHAP Dashboard](assets/shap-dashboard.png)

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run full training + export pipeline.

```bash
python "churn prediction.py"
```

4. Rebuild only the unified report.

```bash
python "churn prediction.py" --build-report-only
```

5. Launch threshold simulator app.

```bash
python "churn prediction.py" --run-threshold-app
```

## Artifacts Included

- churn prediction.py
- threshold_dashboard_app.py
- model_artifacts_report.html
- model_dashboard_plotly.html
- kfold_dashboard_plotly.html
- shap_feature_importance_plotly.html
- metrics_summary.json
- training_history.csv
- threshold_sweep.csv
- kfold_metrics.csv
- shap_feature_importance.csv
- test_predictions.csv
- best_model.keras

## GitHub Push Syntax

```bash
cd "C:/Users/Ebi_Mahmdli/Desktop/churn-prediction-model"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If origin already exists:

```bash
git remote set-url origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Release

Planned release: v1.0.0

This release contains:

- Full churn model pipeline
- Plotly dashboards and SHAP outputs
- Dash threshold app
- Unified artifact report
- Documentation and CI workflow

## Notes

- If customer_data.csv is missing, synthetic data is generated automatically.
- On native Windows with TensorFlow >= 2.11, training usually runs on CPU unless using WSL2 or DirectML.
