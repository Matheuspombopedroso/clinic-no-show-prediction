"""Train the XGBoost no-show classifier and save a versioned artifact.

Steps:

1. Load the synthetic CSV and drop cancelled appointments (a cancelled slot is
   neither a show nor a no-show; cancellations only feed the history features).
2. Split by time: the most recent 20% of appointments (by ``scheduled_at``)
   form the test set. A random split would let the model train on the future
   and evaluate on the past, which overstates real-world performance.
3. Fit ``XGBClassifier`` with early stopping on a validation slice taken from
   the *end* of the training period (again respecting time order).
4. Report ROC-AUC (ranking), PR-AUC (ranking under class imbalance) and Brier
   score (calibration), plus a reliability table, then save everything.

Run: ``python -m noshow.train --data data/appointments.csv --version 1.0.0``
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from noshow.features import FEATURE_COLUMNS, build_features
from noshow.model import save_artifact

TEST_FRACTION = 0.2
VALID_FRACTION = 0.1  # of the training period, used for early stopping

# Moderate, regularised settings: the data set is small and tabular, so
# shallow trees with a low learning rate generalise better than deep ones.
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "early_stopping_rounds": 50,
    "tree_method": "hist",
    "random_state": 42,
}


def time_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into (past, future) by ``scheduled_at``.

    Args:
        df: Appointments with a ``scheduled_at`` column.
        test_fraction: Share of the most recent rows that go to the test set.

    Returns:
        ``(train, test)`` where every test appointment is scheduled at or after
        every train appointment.
    """
    ordered = df.sort_values("scheduled_at", kind="stable").reset_index(drop=True)
    cut = int(len(ordered) * (1.0 - test_fraction))
    return ordered.iloc[:cut], ordered.iloc[cut:]


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Mean predicted vs observed no-show rate per probability decile.

    A well-calibrated model has ``mean_predicted`` close to ``observed_rate``
    in every bin, which is what lets the clinic trust "0.30" to mean ~30%.
    """
    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")
    )
    return [
        {
            "n": int(row["n"]),
            "mean_predicted": round(float(row["mean_predicted"]), 4),
            "observed_rate": round(float(row["observed_rate"]), 4),
        }
        for row in grouped.to_dict("records")
    ]


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Compute the evaluation metrics reported in the README."""
    base_rate = float(np.mean(y_true))
    return {
        "n": len(y_true),
        "no_show_rate": round(base_rate, 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        # Brier of always predicting the base rate: the "no skill" reference.
        "brier_baseline": round(base_rate * (1.0 - base_rate), 4),
        "reliability": reliability_table(y_true, y_prob),
    }


def train(data_path: Path, version: str, out_root: Path) -> dict[str, Any]:
    """Train, evaluate and save a model; return the metadata that was saved."""
    df = pd.read_csv(data_path, parse_dates=["booked_at", "scheduled_at"])
    df = df[df["status"] != "cancelled"]

    train_df, test_df = time_split(df, TEST_FRACTION)
    fit_df, valid_df = time_split(train_df, VALID_FRACTION)

    x_fit, y_fit = build_features(fit_df), fit_df["no_show"].to_numpy()
    x_valid, y_valid = build_features(valid_df), valid_df["no_show"].to_numpy()
    x_test, y_test = build_features(test_df), test_df["no_show"].to_numpy()

    clf = xgb.XGBClassifier(**XGB_PARAMS)
    clf.fit(x_fit, y_fit, eval_set=[(x_valid, y_valid)], verbose=False)

    test_prob = clf.predict_proba(x_test)[:, 1]
    metrics = evaluate(y_test, test_prob)

    metadata: dict[str, Any] = {
        "version": version,
        "model_type": "xgboost.XGBClassifier",
        "features": FEATURE_COLUMNS,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_period": [str(train_df["scheduled_at"].min()), str(train_df["scheduled_at"].max())],
        "test_period": [str(test_df["scheduled_at"].min()), str(test_df["scheduled_at"].max())],
        "n_train": len(train_df),
        "best_iteration": int(clf.best_iteration),
        "params": {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"},
        "test_metrics": metrics,
        "feature_importance_gain": {
            k: round(float(v), 3)
            for k, v in sorted(
                clf.get_booster().get_score(importance_type="gain").items(),
                key=lambda kv: -kv[1],
            )
        },
        "environment": {"python": platform.python_version(), "xgboost": xgb.__version__},
    }
    out_dir = save_artifact(clf.get_booster(), metadata, out_root / version)
    print(f"saved model {version} to {out_dir}")
    return metadata


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the no-show XGBoost model.")
    parser.add_argument("--data", type=Path, default=Path("data/appointments.csv"))
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/no-show-xgb"))
    args = parser.parse_args()

    metadata = train(args.data, args.version, args.out_root)
    summary = {k: v for k, v in metadata["test_metrics"].items() if k != "reliability"}
    print(json.dumps(summary, indent=2))
    print("reliability (test set, by predicted-probability decile):")
    print(pd.DataFrame(metadata["test_metrics"]["reliability"]).to_string(index=False))


if __name__ == "__main__":
    main()
