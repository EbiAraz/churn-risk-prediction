import os
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
