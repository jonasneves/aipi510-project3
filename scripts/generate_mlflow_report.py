#!/usr/bin/env python3
"""Generate static MLflow training report for GitHub Pages."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MLflow HTML summary")
    parser.add_argument("--mlruns", default="mlruns", help="Directory containing MLflow runs")
    parser.add_argument("--experiment", default="salary_prediction", help="MLflow experiment name")
    parser.add_argument("--model-dir", default="models", help="Directory with model artifacts")
    parser.add_argument("--train-log", default="train.log", help="Training log file")
    parser.add_argument("--output", default="docs/mlflow", help="Output directory for report")
    parser.add_argument("--max-log-lines", type=int, default=40, help="Number of log lines to embed")
    return parser.parse_args()


def load_latest_run(tracking_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return None, None

    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return experiment, None
    return experiment, runs[0]


def _format_ts(ts_ms: int) -> str:
    if ts_ms is None:
        return "N/A"
    return datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%SZ")


def partition_metrics(metrics: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    buckets = {"train": {}, "val": {}, "test": {}, "other": {}}
    for key, value in metrics.items():
        if key.startswith("train_"):
            buckets["train"][key.replace("train_", "")] = value
        elif key.startswith("val_"):
            buckets["val"][key.replace("val_", "")] = value
        elif key.startswith("test_"):
            buckets["test"][key.replace("test_", "")] = value
        else:
            buckets["other"][key] = value
    return buckets["train"], buckets["val"], buckets["test"], buckets["other"]


def render_metric_table(title: str, metrics: Dict[str, float]) -> str:
    if not metrics:
        return f"<p>No {title.lower()} metrics logged.</p>"

    rows = "\n".join(
        f"<tr><td>{name}</td><td>{value:,.4f}</td></tr>" for name, value in sorted(metrics.items())
    )
    return f"""
    <h3>{title} Metrics</h3>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """


def render_params_table(params: Dict[str, str]) -> str:
    if not params:
        return "<p>No parameters logged.</p>"
    rows = "\n".join(
        f"<tr><td>{name}</td><td>{value}</td></tr>" for name, value in sorted(params.items())
    )
    return f"""
    <table>
      <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """


def render_feature_table(feature_path: Path) -> Tuple[str, list]:
    if not feature_path.exists():
        return "<p>No feature importance file found.</p>", []

    df = pd.read_csv(feature_path)
    if df.empty or "feature" not in df.columns:
        return "<p>Feature importance file is empty.</p>", []

    df_sorted = df.sort_values("importance", ascending=False)
    rows = "\n".join(
        f"<tr><td>{row.feature}</td><td>{row.importance:.4f}</td></tr>"
        for _, row in df_sorted.head(20).iterrows()
    )
    table_html = f"""
    <table>
      <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """
    return table_html, df_sorted.head(20).to_dict(orient="records")


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"file:{Path(args.mlruns).resolve()}"
    experiment, run = load_latest_run(tracking_uri, args.experiment)

    report_data = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "experiment": args.experiment,
        "tracking_uri": tracking_uri,
        "run": None,
    }

    if not experiment or not run:
        status = "No MLflow experiment found." if not experiment else "No runs found for experiment."
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset='utf-8'>
            <title>MLflow Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
            </style>
        </head>
        <body>
            <h1>MLflow Report</h1>
            <p>{status}</p>
        </body>
        </html>
        """
        (output_dir / "index.html").write_text(html)
        (output_dir / "report.json").write_text(json.dumps(report_data, indent=2))
        return

    report_data["run"] = {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "start_time": _format_ts(run.info.start_time),
        "end_time": _format_ts(run.info.end_time),
        "user": run.info.user_id,
        "params": run.data.params,
        "metrics": run.data.metrics,
    }

    train_metrics, val_metrics, test_metrics, other_metrics = partition_metrics(run.data.metrics)
    params_table = render_params_table(run.data.params)

    # Feature importance
    feature_path = Path(args.model_dir) / "feature_importance.csv"
    feature_table_html, feature_records = render_feature_table(feature_path)
    report_data["feature_importance"] = feature_records

    # Last log lines
    log_text = ""
    tail_lines = []
    train_log_path = Path(args.train_log)
    if train_log_path.exists():
        log_text = train_log_path.read_text()
        tail_lines = log_text.strip().splitlines()[-args.max_log_lines :]
        shutil.copy(train_log_path, output_dir / "train.log")

    metrics_sections = "\n".join(
        [
            render_metric_table("Training", train_metrics),
            render_metric_table("Validation", val_metrics),
            render_metric_table("Test", test_metrics),
        ]
    )

    other_metrics_html = render_metric_table("Other", other_metrics) if other_metrics else ""

    log_section = ""
    if tail_lines:
        log_section = """
        <h2>Recent Training Log</h2>
        <pre>{}</pre>
        """.format("\n".join(tail_lines))

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='utf-8'>
        <title>MLflow Training Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; color: #222; }}
            h1 {{ color: #0b5fff; }}
            h2 {{ margin-top: 2em; color: #0b5fff; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background: #f5f5f5; }}
            pre {{ background: #f7f7f7; padding: 15px; border-radius: 6px; overflow-x: auto; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; }}
            .card {{ border: 1px solid #e0e0e0; border-radius: 6px; padding: 15px; background: #fafafa; }}
        </style>
    </head>
    <body>
        <h1>MLflow Training Report</h1>
        <p><strong>Experiment:</strong> {experiment.name} &middot; <strong>Run ID:</strong> {run.info.run_id}</p>
        <div class="grid">
            <div class="card"><strong>Status</strong><br>{run.info.status}</div>
            <div class="card"><strong>Start</strong><br>{_format_ts(run.info.start_time)}</div>
            <div class="card"><strong>End</strong><br>{_format_ts(run.info.end_time)}</div>
            <div class="card"><strong>User</strong><br>{run.info.user_id or 'N/A'}</div>
        </div>

        <h2>Parameters</h2>
        {params_table}

        <h2>Metrics</h2>
        {metrics_sections}
        {other_metrics_html}

        <h2>Feature Importance</h2>
        {feature_table_html}

        {log_section}
    </body>
    </html>
    """

    (output_dir / "index.html").write_text(html)
    (output_dir / "report.json").write_text(json.dumps(report_data, indent=2))


if __name__ == "__main__":
    main()
