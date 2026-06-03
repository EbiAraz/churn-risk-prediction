import os
import json
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import tensorflow as tf
from plotly.subplots import make_subplots
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class AppConfig:
    seed: int = 42
    data_path: str = os.path.join(BASE_DIR, "customer_data.csv")
    best_model_path: str = os.path.join(BASE_DIR, "best_model.keras")
    predictions_csv_path: str = os.path.join(BASE_DIR, "test_predictions.csv")
    plotly_html_path: str = os.path.join(BASE_DIR, "model_dashboard_plotly.html")
    metrics_json_path: str = os.path.join(BASE_DIR, "metrics_summary.json")
    history_csv_path: str = os.path.join(BASE_DIR, "training_history.csv")
    threshold_csv_path: str = os.path.join(BASE_DIR, "threshold_sweep.csv")
    kfold_csv_path: str = os.path.join(BASE_DIR, "kfold_metrics.csv")
    kfold_html_path: str = os.path.join(BASE_DIR, "kfold_dashboard_plotly.html")
    shap_csv_path: str = os.path.join(BASE_DIR, "shap_feature_importance.csv")
    shap_html_path: str = os.path.join(BASE_DIR, "shap_feature_importance_plotly.html")
    dash_app_path: str = os.path.join(BASE_DIR, "threshold_dashboard_app.py")
    unified_report_path: str = os.path.join(BASE_DIR, "model_artifacts_report.html")
    batch_size: int = 64
    epochs: int = 80
    kfold_splits: int = 5
    kfold_epochs: int = 25
    kfold_patience: int = 4
    shap_background_size: int = 80
    shap_sample_size: int = 200
    shap_nsamples: int = 120


CONFIG = AppConfig()


def configure_runtime(seed: int = CONFIG.seed) -> bool:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    pio.templates.default = "plotly_white"

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU detected: {len(gpus)}")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        for gpu in gpus:
            print(f" - {gpu}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        return True
    else:
        print("No GPU detected. Training will run on CPU.")
        tf.keras.mixed_precision.set_global_policy("float32")
        return False


def load_dataset(data_path: str = CONFIG.data_path) -> pd.DataFrame:
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded dataset: {data_path} | shape={df.shape}")
    else:
        print(f"{data_path} not found. Creating synthetic dataset...")
        x_syn, y_syn = make_classification(
            n_samples=3000,
            n_features=8,
            n_informative=6,
            n_redundant=2,
            n_classes=2,
            weights=[0.7, 0.3],
            random_state=CONFIG.seed,
        )
        columns = [f"feature_{i + 1}" for i in range(8)]
        df = pd.DataFrame(x_syn, columns=columns)
        df["churned"] = y_syn

    if "churned" not in df.columns:
        raise ValueError("Dataset must contain a 'churned' target column")
    return df


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="input_layer"),
            tf.keras.layers.Dense(64, activation="relu", name="hidden_layer_1"),
            tf.keras.layers.BatchNormalization(name="batch_norm_1"),
            tf.keras.layers.Dropout(0.25, name="dropout_1"),
            tf.keras.layers.Dense(32, activation="relu", name="hidden_layer_2"),
            tf.keras.layers.BatchNormalization(name="batch_norm_2"),
            tf.keras.layers.Dropout(0.2, name="dropout_2"),
            tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="output_layer"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
        jit_compile=True,
    )
    return model


def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(buffer_size=min(len(x), 8192), seed=CONFIG.seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def find_best_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_values = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_idx])
    return best_threshold, precisions, recalls, thresholds, f1_values


def compute_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.int32)
        rows.append(
            {
                "threshold": float(thr),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def build_dashboard(
    history: tf.keras.callbacks.History,
    cm: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    precision_curve: np.ndarray,
    recall_curve: np.ndarray,
    threshold_metrics: pd.DataFrame,
    best_threshold: float,
) -> go.Figure:
    epochs = np.arange(1, len(history.history["loss"]) + 1)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Training vs Validation Loss",
            "Training vs Validation Accuracy",
            "ROC Curve",
            "Confusion Matrix (Tuned Threshold)",
            "Precision-Recall Curve",
            "Threshold Sweep (Acc/Prec/Rec/F1)",
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "heatmap"}, {"type": "xy"}, {"type": "xy"}],
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    fig.add_trace(go.Scatter(x=epochs, y=history.history["loss"], name="train_loss", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history.history["val_loss"], name="val_loss", mode="lines"), row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=history.history["accuracy"], name="train_accuracy", mode="lines"), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history.history["val_accuracy"], name="val_accuracy", mode="lines"), row=1, col=2)

    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, name=f"ROC AUC = {roc_auc:.4f}", mode="lines", line=dict(width=3)), row=1, col=3
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], name="random_classifier", mode="lines", line=dict(dash="dash")), row=1, col=3
    )

    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=["Predicted: 0", "Predicted: 1"],
            y=["Actual: 0", "Actual: 1"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            name="confusion_matrix",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=recall_curve, y=precision_curve, name="PR curve", mode="lines", line=dict(width=3, color="#2A9D8F")),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=threshold_metrics["threshold"],
            y=threshold_metrics["accuracy"],
            name="accuracy_by_threshold",
            mode="lines",
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=threshold_metrics["threshold"],
            y=threshold_metrics["precision"],
            name="precision_by_threshold",
            mode="lines",
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=threshold_metrics["threshold"],
            y=threshold_metrics["recall"],
            name="recall_by_threshold",
            mode="lines",
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=threshold_metrics["threshold"],
            y=threshold_metrics["f1"],
            name="f1_by_threshold",
            mode="lines",
            line=dict(width=3, color="#D1495B"),
        ),
        row=2,
        col=3,
    )

    fig.add_vline(
        x=best_threshold,
        line_dash="dash",
        line_width=2,
        line_color="#264653",
        annotation_text=f"best={best_threshold:.3f}",
        annotation_position="top",
        row=2,
        col=3,
    )

    fig.update_layout(
        title="Model Evaluation Dashboard (Interactive Plotly)",
        height=920,
        width=1500,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
        margin=dict(l=40, r=30, t=90, b=60),
    )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=3)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=3)
    fig.update_xaxes(title_text="Recall", row=2, col=2)
    fig.update_yaxes(title_text="Precision", row=2, col=2)
    fig.update_xaxes(title_text="Threshold", row=2, col=3)
    fig.update_yaxes(title_text="Score", row=2, col=3)
    return fig


def save_outputs(
    history: tf.keras.callbacks.History,
    threshold_metrics: pd.DataFrame,
    metrics_summary: dict[str, object],
) -> None:
    pd.DataFrame(history.history).to_csv(CONFIG.history_csv_path, index=False)
    threshold_metrics.to_csv(CONFIG.threshold_csv_path, index=False)
    with open(CONFIG.metrics_json_path, "w", encoding="utf-8") as fp:
        json.dump(metrics_summary, fp, indent=2)


def build_kfold_figure(kfold_df: pd.DataFrame) -> go.Figure:
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc", "pr_auc"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Fold-by-Fold Metrics", "Metric Distribution"),
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.14,
    )

    for metric in metric_cols:
        fig.add_trace(
            go.Scatter(
                x=kfold_df["fold"],
                y=kfold_df[metric],
                mode="lines+markers",
                name=f"{metric}_by_fold",
            ),
            row=1,
            col=1,
        )

    means = kfold_df[metric_cols].mean()
    stds = kfold_df[metric_cols].std().fillna(0.0)
    fig.add_trace(
        go.Bar(
            x=metric_cols,
            y=means.values,
            error_y=dict(type="data", array=stds.values, visible=True),
            name="mean±std",
            marker_color="#2A9D8F",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="K-Fold Cross-Validation Performance (Neural Net)",
        height=520,
        width=1250,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=70, b=40),
    )
    fig.update_xaxes(title_text="Fold", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_xaxes(title_text="Metric", row=1, col=2)
    fig.update_yaxes(title_text="Mean Score", row=1, col=2)
    return fig


def run_kfold_cross_validation(x: np.ndarray, y: np.ndarray, has_gpu: bool) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=CONFIG.kfold_splits, shuffle=True, random_state=CONFIG.seed)
    rows: list[dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        print(f"K-Fold {fold_idx}/{CONFIG.kfold_splits}: training...")
        x_train_fold, x_test_fold = x[train_idx], x[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        x_train_fold = scaler.fit_transform(x_train_fold).astype(np.float32)
        x_test_fold = scaler.transform(x_test_fold).astype(np.float32)

        x_fit, x_val, y_fit, y_val = train_test_split(
            x_train_fold,
            y_train_fold,
            test_size=0.2,
            random_state=CONFIG.seed + fold_idx,
            stratify=y_train_fold,
        )

        class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_fit)
        class_weight = {0: float(class_weights[0]), 1: float(class_weights[1])}

        model = build_model(input_dim=x_train_fold.shape[1])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=CONFIG.kfold_patience,
                restore_best_weights=True,
                verbose=0,
            ),
        ]

        train_ds = make_tf_dataset(x_fit, y_fit, batch_size=CONFIG.batch_size, training=True)
        val_ds = make_tf_dataset(x_val, y_val, batch_size=CONFIG.batch_size, training=False)
        test_ds = make_tf_dataset(x_test_fold, y_test_fold, batch_size=CONFIG.batch_size, training=False)

        model.fit(
            train_ds,
            epochs=CONFIG.kfold_epochs,
            validation_data=val_ds,
            verbose=0,
            callbacks=callbacks,
            class_weight=class_weight,
        )

        _, _, _, _, fold_auc, fold_pr_auc = model.evaluate(test_ds, verbose=0)
        y_prob = model.predict(test_ds, verbose=0).ravel()
        best_threshold, _, _, _, _ = find_best_threshold(y_test_fold, y_prob)
        y_pred = (y_prob >= best_threshold).astype(np.int32)

        rows.append(
            {
                "fold": float(fold_idx),
                "accuracy": float(accuracy_score(y_test_fold, y_pred)),
                "precision": float(precision_score(y_test_fold, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test_fold, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test_fold, y_pred, zero_division=0)),
                "auc": float(fold_auc),
                "pr_auc": float(fold_pr_auc),
                "best_threshold": float(best_threshold),
                "has_gpu": float(has_gpu),
            }
        )

    kfold_df = pd.DataFrame(rows)
    kfold_df.to_csv(CONFIG.kfold_csv_path, index=False)
    kfold_fig = build_kfold_figure(kfold_df)
    kfold_fig.write_html(CONFIG.kfold_html_path)
    return kfold_df


def run_shap_analysis(
    model: tf.keras.Model,
    x_train_scaled: np.ndarray,
    x_test_scaled: np.ndarray,
    feature_columns: list[str],
) -> tuple[bool, str]:
    try:
        import shap
    except ImportError:
        return False, "SHAP is not installed. Install with: pip install shap"

    rng = np.random.default_rng(CONFIG.seed)
    bg_size = min(CONFIG.shap_background_size, len(x_train_scaled))
    sample_size = min(CONFIG.shap_sample_size, len(x_test_scaled))

    bg_idx = rng.choice(len(x_train_scaled), size=bg_size, replace=False)
    sample_idx = rng.choice(len(x_test_scaled), size=sample_size, replace=False)

    background = x_train_scaled[bg_idx]
    sample = x_test_scaled[sample_idx]

    def predict_fn(data: np.ndarray) -> np.ndarray:
        return model.predict(np.asarray(data, dtype=np.float32), verbose=0).ravel()

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(sample, nsamples=CONFIG.shap_nsamples)
    shap_values = np.asarray(shap_values)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_columns, "mean_abs_shap": mean_abs_shap})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df.to_csv(CONFIG.shap_csv_path, index=False)

    shap_fig = go.Figure(
        data=[
            go.Bar(
                x=shap_df["mean_abs_shap"],
                y=shap_df["feature"],
                orientation="h",
                marker_color="#E76F51",
                name="mean_abs_shap",
            )
        ]
    )
    shap_fig.update_layout(
        title="SHAP Feature Importance (Mean |SHAP|)",
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        height=520,
        width=900,
        margin=dict(l=100, r=30, t=70, b=50),
    )
    shap_fig.write_html(CONFIG.shap_html_path)

    return True, "SHAP analysis completed"


def export_threshold_dash_app() -> None:
    app_code = '''import os
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_CSV = os.path.join(BASE_DIR, "test_predictions.csv")
THRESHOLD_CSV = os.path.join(BASE_DIR, "threshold_sweep.csv")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = pd.read_csv(PREDICTIONS_CSV)
    thresh = pd.read_csv(THRESHOLD_CSV)
    return preds, thresh


def cm_figure(cm):
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=cm,
                x=["Predicted: 0", "Predicted: 1"],
                y=["Actual: 0", "Actual: 1"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        ]
    )
    fig.update_layout(title="Confusion Matrix", height=420, width=560)
    return fig


def threshold_curve_figure(thresh_df, active_threshold):
    fig = go.Figure()
    for col in ["accuracy", "precision", "recall", "f1"]:
        fig.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df[col], mode="lines", name=col))

    fig.add_vline(
        x=active_threshold,
        line_dash="dash",
        line_color="#264653",
        annotation_text=f"threshold={active_threshold:.3f}",
        annotation_position="top",
    )
    fig.update_layout(title="Threshold Metrics", height=420, width=760, yaxis_title="Score")
    return fig


predictions_df, threshold_df = load_data()

app = Dash(__name__)
app.layout = html.Div(
    [
        html.H2("Interactive Threshold Simulator"),
        dcc.Slider(
            id="threshold-slider",
            min=0.01,
            max=0.99,
            step=0.01,
            value=0.5,
            marks={i / 10: f"{i / 10:.1f}" for i in range(1, 10)},
        ),
        html.Div(id="metric-cards", style={"margin": "14px 0", "fontSize": "18px"}),
        html.Div(
            [
                dcc.Graph(id="cm-graph", style={"display": "inline-block", "verticalAlign": "top"}),
                dcc.Graph(id="threshold-graph", style={"display": "inline-block"}),
            ]
        ),
    ],
    style={"fontFamily": "Segoe UI", "padding": "14px"},
)


@app.callback(
    Output("metric-cards", "children"),
    Output("cm-graph", "figure"),
    Output("threshold-graph", "figure"),
    Input("threshold-slider", "value"),
)
def update_dashboard(threshold):
    y_true = predictions_df["actual_churned"].to_numpy()
    y_prob = predictions_df["predicted_probability"].to_numpy()
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    tp = int(cm[1, 1])
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    text = (
        f"Accuracy={accuracy:.4f} | Precision={precision:.4f} | "
        f"Recall={recall:.4f} | F1={f1:.4f} | TP={tp} TN={tn} FP={fp} FN={fn}"
    )
    return text, cm_figure(cm), threshold_curve_figure(threshold_df, threshold)


if __name__ == "__main__":
    app.run(debug=False)
'''
    with open(CONFIG.dash_app_path, "w", encoding="utf-8") as fp:
        fp.write(app_code)


def _report_row_html(label: str, path: str, description: str) -> str:
    exists = os.path.exists(path)
    status = "Available" if exists else "Missing"
    status_color = "#2A9D8F" if exists else "#D1495B"
    rel = os.path.basename(path)
    link_html = f'<a href="{rel}" target="_blank">{rel}</a>' if exists else rel
    return (
        "<tr>"
        f"<td>{label}</td>"
        f"<td style='color:{status_color};font-weight:600'>{status}</td>"
        f"<td>{link_html}</td>"
        f"<td>{description}</td>"
        "</tr>"
    )


def export_unified_report(metrics_summary: dict[str, object] | None = None) -> None:
    if metrics_summary is None and os.path.exists(CONFIG.metrics_json_path):
        with open(CONFIG.metrics_json_path, "r", encoding="utf-8") as fp:
            metrics_summary = json.load(fp)

    metrics_summary = metrics_summary or {}

    metric_lines = []
    metric_keys = [
        "baseline_accuracy",
        "baseline_auc",
        "baseline_pr_auc",
        "best_threshold",
        "tuned_accuracy",
        "tuned_precision",
        "tuned_recall",
        "tuned_f1",
    ]
    for key in metric_keys:
        value = metrics_summary.get(key)
        if isinstance(value, (float, int)):
            metric_lines.append(f"<li><b>{key}</b>: {float(value):.4f}</li>")

    if not metric_lines:
        metric_lines.append("<li>No metrics found yet. Run training first.</li>")

    artifacts = [
        ("Main Dashboard", CONFIG.plotly_html_path, "Interactive training + ROC + threshold visualizations"),
        ("K-Fold Dashboard", CONFIG.kfold_html_path, "Cross-validation fold and distribution view"),
        ("SHAP Dashboard", CONFIG.shap_html_path, "Feature importance explainability chart"),
        ("Threshold App", CONFIG.dash_app_path, "Interactive Dash app with threshold slider"),
        ("Predictions CSV", CONFIG.predictions_csv_path, "Final test predictions and labels"),
        ("Metrics JSON", CONFIG.metrics_json_path, "Summary metrics in machine-readable format"),
        ("Threshold Sweep CSV", CONFIG.threshold_csv_path, "Metrics at different decision thresholds"),
        ("Training History CSV", CONFIG.history_csv_path, "Epoch-level model training metrics"),
        ("K-Fold CSV", CONFIG.kfold_csv_path, "Per-fold CV metrics"),
        ("SHAP CSV", CONFIG.shap_csv_path, "Mean absolute SHAP values by feature"),
        ("Best Model", CONFIG.best_model_path, "Best checkpoint saved by validation loss"),
    ]

    dashboard_paths = [
        ("Main Dashboard", CONFIG.plotly_html_path),
        ("K-Fold Dashboard", CONFIG.kfold_html_path),
        ("SHAP Dashboard", CONFIG.shap_html_path),
    ]
    dashboard_links = [
        (label, os.path.basename(path), os.path.exists(path)) for label, path in dashboard_paths
    ]

    button_parts = []
    for label, rel, exists in dashboard_links:
        if exists:
            button_parts.append(f'<a class="launch-btn" href="{rel}" target="_blank">Open {label}</a>')
        else:
            button_parts.append(f'<span class="launch-btn disabled">Open {label} (Missing)</span>')
    dashboard_buttons_html = "".join(button_parts)

    available_dash_links = [f'"{rel}"' for _, rel, exists in dashboard_links if exists]
    open_all_js_array = f"[{','.join(available_dash_links)}]"

    rows_html = "\n".join(_report_row_html(name, path, desc) for name, path, desc in artifacts)
    report_html = f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Model Artifacts Report</title>
    <style>
        :root {{
            --bg: #f7f6f2;
            --card: #ffffff;
            --ink: #1f2937;
            --muted: #5b6470;
            --line: #d8dee8;
            --brand: #264653;
            --accent: #2a9d8f;
        }}
        body {{ margin: 0; font-family: Segoe UI, Tahoma, sans-serif; background: linear-gradient(130deg, #f7f6f2, #e6f2ef); color: var(--ink); }}
        .container {{ max-width: 1100px; margin: 0 auto; padding: 28px 18px 44px; }}
        .hero {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 20px; box-shadow: 0 8px 20px rgba(38,70,83,0.08); }}
        .hero h1 {{ margin: 0 0 8px; color: var(--brand); }}
        .hero p {{ margin: 0; color: var(--muted); }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
        .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px; }}
        .card h2 {{ margin: 0 0 10px; font-size: 18px; color: var(--brand); }}
        .card ul {{ margin: 0; padding-left: 20px; }}
        .button-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }}
        .launch-btn {{
            display: inline-block;
            padding: 10px 14px;
            border-radius: 10px;
            border: 1px solid #d0d8e4;
            background: #ecf3f8;
            color: var(--brand);
            font-weight: 700;
            text-decoration: none;
            cursor: pointer;
        }}
        .launch-btn:hover {{ background: #dde9f2; text-decoration: none; }}
        .launch-btn.disabled {{ background: #f3f4f6; color: #8a94a6; border-color: #e0e4eb; cursor: not-allowed; }}
        a {{ color: var(--brand); text-decoration: none; font-weight: 600; }}
        a:hover {{ text-decoration: underline; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14px; }}
        th, td {{ border-bottom: 1px solid var(--line); text-align: left; padding: 10px 8px; vertical-align: top; }}
        th {{ color: var(--muted); font-weight: 700; }}
        .footer {{ margin-top: 14px; color: var(--muted); font-size: 13px; }}
        @media (max-width: 840px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class=\"container\">
        <section class=\"hero\">
            <h1>Unified Model Artifacts Report</h1>
            <p>Single entry point for dashboards, exported files, and quick run commands.</p>
        </section>

        <section class=\"grid\">
            <article class=\"card\">
                <h2>Key Metrics</h2>
                <ul>
                    {''.join(metric_lines)}
                </ul>
            </article>
            <article class=\"card\">
                <h2>Quick Commands</h2>
                <ul>
                    <li>Run training pipeline: python Confusion-Matrix.py</li>
                    <li>Run threshold app: python Confusion-Matrix.py --run-threshold-app</li>
                    <li>Rebuild this report only: python Confusion-Matrix.py --build-report-only</li>
                </ul>
            </article>
        </section>

        <section class=\"card\" style=\"margin-top:16px\">
            <h2>Dashboard Launch</h2>
            <div class=\"button-row\">
                <button class=\"launch-btn\" onclick=\"openAllDashboards()\">Open All Dashboards</button>
                {dashboard_buttons_html}
            </div>
        </section>

        <section class=\"card\" style=\"margin-top:16px\">
            <h2>Artifacts</h2>
            <table>
                <thead>
                    <tr><th>Item</th><th>Status</th><th>File</th><th>Description</th></tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            <div class=\"footer\">Generated by Confusion-Matrix.py in the same directory as this report.</div>
        </section>
    </div>
    <script>
        function openAllDashboards() {{
        const links = {open_all_js_array};
        if (!links.length) {{
            alert("No dashboard files were found yet. Run training first.");
            return;
        }}
        links.forEach((url) => window.open(url, "_blank"));
        }}
    </script>
</body>
</html>
"""
    with open(CONFIG.unified_report_path, "w", encoding="utf-8") as fp:
        fp.write(report_html)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn model with Plotly dashboard, K-Fold and SHAP analysis")
    parser.add_argument("--run-threshold-app", action="store_true", help="Run generated Dash threshold simulator app")
    parser.add_argument("--build-report-only", action="store_true", help="Build unified artifact report without training")
    return parser.parse_args()


def run_threshold_app() -> None:
    if not os.path.exists(CONFIG.dash_app_path):
        raise FileNotFoundError(
            f"{CONFIG.dash_app_path} is missing. Run training once to generate all app files first."
        )
    with open(CONFIG.dash_app_path, "r", encoding="utf-8") as fp:
        code = compile(fp.read(), CONFIG.dash_app_path, "exec")
        exec(code, {"__name__": "__main__"})


def main() -> None:
    print("=" * 70)
    print("STEP 1/3: SETUP + DATA + TRAIN")
    print("=" * 70)
    has_gpu = configure_runtime()

    df = load_dataset()
    feature_columns = [col for col in df.columns if col != "churned"]

    x = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["churned"].to_numpy(dtype=np.int32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=CONFIG.seed,
        stratify=y,
    )

    print(f"Training set size: {len(x_train)}")
    print(f"Testing set size: {len(x_test)}")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train_scaled,
        y_train,
        test_size=0.2,
        random_state=CONFIG.seed,
        stratify=y_train,
    )

    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_fit)
    class_weight = {0: float(class_weights[0]), 1: float(class_weights[1])}

    model = build_model(input_dim=x_train_scaled.shape[1])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(CONFIG.best_model_path, save_best_only=True, monitor="val_loss", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1),
    ]

    train_ds = make_tf_dataset(x_fit, y_fit, batch_size=CONFIG.batch_size, training=True)
    val_ds = make_tf_dataset(x_val, y_val, batch_size=CONFIG.batch_size, training=False)
    test_ds = make_tf_dataset(x_test_scaled, y_test, batch_size=CONFIG.batch_size, training=False)

    history = model.fit(
        train_ds,
        epochs=CONFIG.epochs,
        validation_data=val_ds,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    print("\nTraining complete.")

    print("=" * 70)
    print("STEP 2/3: EVALUATION + THRESHOLD TUNING + PLOTLY")
    print("=" * 70)

    test_loss, test_accuracy, test_precision, test_recall, test_auc, test_pr_auc = model.evaluate(test_ds, verbose=0)
    print("Baseline metrics @ threshold=0.50")
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"AUC:       {test_auc:.4f}")
    print(f"PR-AUC:    {test_pr_auc:.4f}")

    y_prob = model.predict(test_ds, verbose=0).ravel()
    best_threshold, precisions, recalls, _, _ = find_best_threshold(y_test, y_prob)

    y_pred = (y_prob >= best_threshold).astype(np.int32)
    tuned_acc = accuracy_score(y_test, y_pred)
    tuned_precision = precision_score(y_test, y_pred, zero_division=0)
    tuned_recall = recall_score(y_test, y_pred, zero_division=0)
    tuned_f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nTuned metrics @ best F1 threshold")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Accuracy:       {tuned_acc:.4f}")
    print(f"Precision:      {tuned_precision:.4f}")
    print(f"Recall:         {tuned_recall:.4f}")
    print(f"F1-score:       {tuned_f1:.4f}")

    print("\nClassification Report (tuned threshold):")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    threshold_grid = np.linspace(0.05, 0.95, 37)
    threshold_metrics = compute_threshold_metrics(y_test, y_prob, threshold_grid)

    fig = build_dashboard(
        history,
        cm,
        fpr,
        tpr,
        roc_auc,
        precisions,
        recalls,
        threshold_metrics,
        best_threshold,
    )
    fig.write_html(CONFIG.plotly_html_path)
    fig.show()

    metrics_summary = {
        "has_gpu": has_gpu,
        "best_threshold": float(best_threshold),
        "baseline_loss": float(test_loss),
        "baseline_accuracy": float(test_accuracy),
        "baseline_precision": float(test_precision),
        "baseline_recall": float(test_recall),
        "baseline_auc": float(test_auc),
        "baseline_pr_auc": float(test_pr_auc),
        "tuned_accuracy": float(tuned_acc),
        "tuned_precision": float(tuned_precision),
        "tuned_recall": float(tuned_recall),
        "tuned_f1": float(tuned_f1),
        "roc_auc": float(roc_auc),
    }
    save_outputs(history, threshold_metrics, metrics_summary)

    print(f"\nInteractive dashboard exported: {CONFIG.plotly_html_path}")
    print(f"Training history exported: {CONFIG.history_csv_path}")
    print(f"Threshold sweep exported: {CONFIG.threshold_csv_path}")
    print(f"Metrics summary exported: {CONFIG.metrics_json_path}")

    print("=" * 70)
    print("STEP 3/3: EXPORT PREDICTIONS CSV")
    print("=" * 70)

    predictions_df = pd.DataFrame(x_test, columns=feature_columns)
    predictions_df["actual_churned"] = y_test
    predictions_df["predicted_probability"] = y_prob
    predictions_df["predicted_churned"] = y_pred
    predictions_df.to_csv(CONFIG.predictions_csv_path, index=False)

    print(f"Predictions exported: {CONFIG.predictions_csv_path}")
    print(predictions_df.head(10).to_string(index=False))

    print("=" * 70)
    print("STEP 4/5: K-FOLD CROSS-VALIDATION + EXPORT")
    print("=" * 70)
    kfold_df = run_kfold_cross_validation(x, y, has_gpu=has_gpu)
    print(f"K-Fold metrics exported: {CONFIG.kfold_csv_path}")
    print(f"K-Fold dashboard exported: {CONFIG.kfold_html_path}")
    print(kfold_df.describe().to_string())

    print("=" * 70)
    print("STEP 5/5: SHAP EXPLAINABILITY + DASH APP EXPORT")
    print("=" * 70)
    shap_ok, shap_msg = run_shap_analysis(model, x_train_scaled, x_test_scaled, feature_columns)
    print(shap_msg)
    if shap_ok:
        print(f"SHAP importance CSV exported: {CONFIG.shap_csv_path}")
        print(f"SHAP importance chart exported: {CONFIG.shap_html_path}")

    export_threshold_dash_app()
    print(f"Threshold Dash app generated: {CONFIG.dash_app_path}")

    export_unified_report(metrics_summary)
    print(f"Unified report generated: {CONFIG.unified_report_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.run_threshold_app:
        run_threshold_app()
    elif args.build_report_only:
        export_unified_report()
        print(f"Unified report generated: {CONFIG.unified_report_path}")
    else:
        main()