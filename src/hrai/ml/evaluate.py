"""Evaluation, calibration and threshold selection (Steps 06-07).

Three deliberate departures from "read the top of the accuracy column":

**Metrics.** Accuracy is excluded entirely. At a 16.1% positive rate a model
that predicts "stays" for everyone scores 83.9%, so accuracy rewards exactly the
behaviour we are trying to avoid. We report precision, recall, F1, ROC-AUC and
PR-AUC (which is the honest one under imbalance).

**Calibration.** Step 16 buckets probabilities into HIGH / MEDIUM / LOW risk
bands, so the numbers must be *correct*, not merely correctly *ordered*. An
uncalibrated model can rank perfectly and band terribly. We report the Brier
score and reliability, and calibrate the winner.

**Threshold.** The Build Notes say to choose on "the actual cost of mistakes".
Made quantitative: a false negative costs a replacement (a multiple of the
employee's own salary), a false positive costs one unnecessary intervention.
The operating threshold is the one that minimises expected cost — not 0.5, and
not whatever maximises F1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from hrai.utils.config import get
from hrai.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier: float
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    support_positive: int
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return Metrics(
        precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        roc_auc=round(float(roc_auc_score(y_true, y_prob)), 4),
        pr_auc=round(float(average_precision_score(y_true, y_prob)), 4),
        brier=round(float(brier_score_loss(y_true, y_prob)), 4),
        threshold=round(float(threshold), 4),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        support_positive=int(y_true.sum()),
        n=int(len(y_true)),
    )


def expected_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_false_negative: np.ndarray,
    cost_false_positive: float,
) -> float:
    """Total expected cost of operating at ``threshold``.

    ``cost_false_negative`` is per-row: losing a senior engineer costs more than
    losing a junior one, and a single global constant would hide that.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    missed = (y_true == 1) & (y_pred == 0)
    false_alarm = (y_true == 0) & (y_pred == 1)
    return float(cost_false_negative[missed].sum() + cost_false_positive * false_alarm.sum())


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    monthly_income: pd.Series | np.ndarray | None = None,
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pick the operating threshold that minimises expected cost.

    Both costs scale with the employee's own salary, so the ratio is constant
    and the theoretical optimum is the classic cost-sensitive Bayes threshold,
    ``cost_fp / (cost_fp + cost_fn)``. We compute that analytically *and* search
    the grid empirically: when the two agree, the search is behaving; when they
    diverge, something is wrong with the probabilities and we want to know.
    """
    cfg = get("model.cost_sensitive", {}) or {}
    fn_multiple = float(cfg.get("false_negative_cost_multiple", 6.0))
    fp_multiple = float(cfg.get("false_positive_cost_multiple", 0.25))
    currency = cfg.get("currency", "INR")

    n = len(y_true)
    if monthly_income is not None:
        income_series = pd.Series(monthly_income).astype(float)
        income = np.asarray(income_series.fillna(income_series.median()))
    else:
        income = np.full(n, 50_000.0)

    cost_fn = income * fn_multiple
    cost_fp_row = income * fp_multiple
    mean_cost_fp = float(np.mean(cost_fp_row))

    ratio = fn_multiple / fp_multiple if fp_multiple else float("inf")
    bayes_threshold = round(fp_multiple / (fp_multiple + fn_multiple), 4)

    # Wide grid: a boundary hit would mean the true optimum lies outside it.
    grid = grid if grid is not None else np.round(np.arange(0.01, 1.00, 0.01), 2)
    costs = [float(_row_cost(y_true, y_prob, t, cost_fn, cost_fp_row)) for t in grid]
    best_index = int(np.argmin(costs))
    best_threshold = float(grid[best_index])
    at_boundary = best_index in (0, len(grid) - 1)

    baseline = float(_row_cost(y_true, y_prob, 0.5, cost_fn, cost_fp_row))
    do_nothing = float(cost_fn[np.asarray(y_true).astype(bool)].sum())

    result = {
        "threshold": best_threshold,
        "bayes_optimal_threshold": bayes_threshold,
        "threshold_at_grid_boundary": at_boundary,
        "cost_ratio_fn_to_fp": round(ratio, 1),
        "expected_cost": round(costs[best_index], 2),
        "expected_cost_at_0.5": round(baseline, 2),
        "cost_saving_vs_0.5": round(baseline - costs[best_index], 2),
        "cost_if_no_model": round(do_nothing, 2),
        "mean_false_positive_cost": round(mean_cost_fp, 2),
        "currency": currency,
        "false_negative_cost_multiple": fn_multiple,
        "false_positive_cost_multiple": fp_multiple,
        "curve": [
            {"threshold": float(t), "cost": round(float(c), 2)}
            for t, c in zip(grid, costs, strict=True)
        ],
    }

    if at_boundary:
        log.warning(
            "cost-optimal threshold sits on the grid boundary — the true optimum may "
            "lie outside the searched range",
            extra={
                "threshold": best_threshold,
                "grid_min": float(grid[0]),
                "grid_max": float(grid[-1]),
            },
        )
    log.info(
        "operating threshold selected by expected cost",
        extra={k: v for k, v in result.items() if k != "curve"},
    )
    return result


def capacity_threshold(y_prob: np.ndarray, capacity_pct: float | None = None) -> dict[str, Any]:
    """The threshold that flags exactly as many people as HR can actually review.

    Cost optimality and operational reality are different questions. With a ~24:1
    cost asymmetry the cost-optimal threshold flags a large share of the
    workforce, which is correct economics and useless as a work queue. This gives
    the dashboard a second, capacity-bounded operating point.
    """
    capacity_pct = (
        capacity_pct
        if capacity_pct is not None
        else float(get("model.cost_sensitive.hr_review_capacity_pct", 0.20))
    )
    y_prob = np.asarray(y_prob, dtype=float)
    k = max(1, int(round(capacity_pct * len(y_prob))))
    threshold = float(np.sort(y_prob)[::-1][k - 1])
    return {
        "capacity_pct": capacity_pct,
        "employees_flagged": k,
        "threshold": round(threshold, 4),
    }


def _row_cost(y_true, y_prob, threshold, cost_fn_row, cost_fp_row) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    missed = (y_true == 1) & (y_pred == 0)
    false_alarm = (y_true == 0) & (y_pred == 1)
    return float(cost_fn_row[missed].sum() + cost_fp_row[false_alarm].sum())


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10
) -> list[dict[str, Any]]:
    """Predicted vs observed rate per probability bucket — the calibration story."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob, dtype=float)})
    df["bucket"] = pd.cut(df["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = (
        df.groupby("bucket", observed=True)
        .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    return [
        {
            "bucket": str(row["bucket"]),
            "n": int(row["n"]),
            "predicted_rate": round(float(row["predicted"]), 4),
            "observed_rate": round(float(row["observed"]), 4),
        }
        for _, row in grouped.iterrows()
    ]


def risk_band(probability: float) -> str:
    """Map a calibrated probability to the HIGH / MEDIUM / LOW band used everywhere."""
    bands = get("model.risk_bands", {"high": 0.6, "medium": 0.3})
    if probability >= float(bands.get("high", 0.6)):
        return "HIGH"
    if probability >= float(bands.get("medium", 0.3)):
        return "MEDIUM"
    return "LOW"


def risk_band_series(probabilities: pd.Series) -> pd.Series:
    return pd.Series([risk_band(float(p)) for p in probabilities], index=probabilities.index)
