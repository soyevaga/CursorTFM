"""Fair ML pipeline for the German Credit dataset (OpenML credit-g).

Loads the data, preprocesses without leakage, trains a class-balanced
Random Forest, mitigates gender disparity with Fairlearn, selects a
decision threshold on out-of-fold train scores, and reports unbiased
test metrics in output_metrics.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OPENML_DATA_ID = 31
TARGET_COL = "class"
POSITIVE_LABEL = "good"
PROTECTED_COL = "Sex"
PERSONAL_STATUS_COL = "personal_status"
PRIVILEGED_GROUP = "male"
UNPRIVILEGED_GROUP = "female"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
MIN_F1 = 0.70
MIN_DI = 0.80
OUTPUT_METRICS_PATH = Path("output_metrics.json")


def load_german_credit(data_id: int = OPENML_DATA_ID) -> pd.DataFrame:
    """Load OpenML credit-g and derive the protected attribute Sex."""
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    df = bunch.frame.copy()

    if PROTECTED_COL not in df.columns:
        if PERSONAL_STATUS_COL not in df.columns:
            raise KeyError(
                "Neither 'Sex' nor 'personal_status' is available."
            )
        status = df[PERSONAL_STATUS_COL].astype(str)
        df[PROTECTED_COL] = status.str.contains(
            "female",
            case=False,
        ).map({True: "female", False: "male"})

    df[PROTECTED_COL] = df[PROTECTED_COL].astype("category")
    df[TARGET_COL] = df[TARGET_COL].astype("category")
    return df


def base_approval_rate_by_group(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    group_col: str = PROTECTED_COL,
    positive_label: str = POSITIVE_LABEL,
) -> pd.DataFrame:
    """Baseline P(Y=positive | group) and sample sizes."""
    approved = df[target_col].eq(positive_label)
    stats = (
        df.assign(_approved=approved)
        .groupby(group_col, observed=True)["_approved"]
        .agg(n="size", n_approved="sum", approval_rate="mean")
        .reset_index()
    )
    stats["approval_rate"] = stats["approval_rate"].astype(float)
    return stats.sort_values(group_col).reset_index(drop=True)


def make_xy_s(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    protected_col: str = PROTECTED_COL,
    leak_cols: Iterable[str] = (PERSONAL_STATUS_COL,),
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split features, label and protected attribute.

    ``personal_status`` is dropped because it encodes Sex. Sex itself is
    kept out of X (fairness through unawareness).
    """
    drop_cols = {target_col, protected_col, *leak_cols}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col].copy()
    s = df[protected_col].copy()
    return X, y, s


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    s: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """80/20 split stratified by the label."""
    return train_test_split(
        X,
        y,
        s,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def _infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Unfitted ColumnTransformer (fit only later, on train / CV folds)."""
    numeric_cols, categorical_cols = _infer_column_types(X_train)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def to_binary(
    y: pd.Series | np.ndarray,
    positive_label: str = POSITIVE_LABEL,
) -> np.ndarray:
    """Map labels to {0, 1} with ``good`` as the positive class."""
    y = pd.Series(y)
    unique_vals = set(pd.unique(y.dropna()))
    if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
        return y.astype(int).to_numpy()
    return y.astype(str).eq(str(positive_label)).astype(int).to_numpy()


def disparate_impact(
    y_pred: np.ndarray,
    sensitive: pd.Series | np.ndarray,
    *,
    privileged: str = PRIVILEGED_GROUP,
    unprivileged: str = UNPRIVILEGED_GROUP,
) -> float:
    """P(Ŷ=1 | unprivileged) / P(Ŷ=1 | privileged)."""
    groups = pd.Series(sensitive).astype(str).to_numpy()
    y_pred = np.asarray(y_pred).astype(int)
    sr_priv = y_pred[groups == privileged].mean()
    sr_unpriv = y_pred[groups == unprivileged].mean()
    if sr_priv == 0:
        return float("nan")
    return float(sr_unpriv / sr_priv)


def build_fair_rf_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Preprocessing + RandomForest with class_weight='balanced'."""
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("pre", build_preprocessor(X_train)),
            ("clf", clf),
        ]
    )


def fit_threshold_optimizer(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    s_train: pd.Series,
) -> ThresholdOptimizer:
    """Group-specific thresholds under demographic parity (Sex)."""
    y_bin = to_binary(y_train)
    post = ThresholdOptimizer(
        estimator=estimator,
        constraints="demographic_parity",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    post.fit(X_train, y_bin, sensitive_features=s_train)
    return post


@dataclass
class ThresholdSearchResult:
    threshold: float
    f1: float
    di: float
    selection_rate: float


def optimize_threshold_f1_di(
    y_true: np.ndarray,
    proba: np.ndarray,
    sensitive: pd.Series | np.ndarray,
    *,
    min_f1: float = MIN_F1,
    min_di: float = MIN_DI,
    grid: np.ndarray | None = None,
) -> ThresholdSearchResult:
    """Pick t on the selection set (OOF/train), never on test."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)

    feasible: list[ThresholdSearchResult] = []
    for threshold in grid:
        y_pred = (proba >= threshold).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        di = disparate_impact(y_pred, sensitive)
        if np.isnan(di) or f1 < min_f1 or di < min_di:
            continue
        feasible.append(
            ThresholdSearchResult(
                threshold=float(threshold),
                f1=f1,
                di=di,
                selection_rate=float(y_pred.mean()),
            )
        )

    if not feasible:
        raise RuntimeError(
            f"No train/OOF threshold meets F1>={min_f1} and DI>={min_di}."
        )

    return max(
        feasible,
        key=lambda row: (row.f1, row.di, -abs(row.threshold - 0.5)),
    )


def oof_predict_proba(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train_bin: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Out-of-fold probabilities; each fold fits pre+RF on fold-train only."""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    proba = cross_val_predict(
        pipeline,
        X_train,
        y_train_bin,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )
    return proba[:, 1]


def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(proba) >= threshold).astype(int)


def compute_test_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    sensitive: pd.Series,
) -> dict[str, float]:
    """Unbiased test metrics for the frozen decision threshold."""
    return {
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "disparate_impact": disparate_impact(y_pred, sensitive),
        "demographic_parity_difference": float(
            demographic_parity_difference(
                y_true,
                y_pred,
                sensitive_features=sensitive,
            )
        ),
    }


def save_metrics(metrics: dict[str, float], path: Path = OUTPUT_METRICS_PATH) -> None:
    payload = {
        "f1_score": float(metrics["f1_score"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "roc_auc": float(metrics["roc_auc"]),
        "disparate_impact": float(metrics["disparate_impact"]),
        "demographic_parity_difference": float(
            metrics["demographic_parity_difference"]
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_fair_credit_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    s_train: pd.Series,
    s_test: pd.Series,
) -> dict:
    """Fit on train, select t via OOF, evaluate the frozen threshold on test."""
    y_train_bin = to_binary(y_train)
    y_test_bin = to_binary(y_test)

    pipe_for_oof = build_fair_rf_pipeline(X_train)
    proba_oof = oof_predict_proba(pipe_for_oof, X_train, y_train_bin)
    chosen = optimize_threshold_f1_di(y_train_bin, proba_oof, s_train)
    threshold = chosen.threshold

    pipe = build_fair_rf_pipeline(X_train)
    pipe.fit(X_train, y_train_bin)
    post = fit_threshold_optimizer(pipe, X_train, y_train, s_train)

    proba_test = pipe.predict_proba(X_test)[:, 1]
    y_pred_test = apply_threshold(proba_test, threshold)
    y_pred_to = post.predict(
        X_test,
        sensitive_features=s_test,
        random_state=RANDOM_STATE,
    ).astype(int)

    return {
        "pipeline": pipe,
        "threshold_optimizer": post,
        "threshold": threshold,
        "oof_metrics": {
            "f1": chosen.f1,
            "di": chosen.di,
            "selection_rate": chosen.selection_rate,
        },
        "test_proba": proba_test,
        "y_pred_threshold": y_pred_test,
        "y_pred_fairlearn_to": y_pred_to,
        "y_test_bin": y_test_bin,
        "metrics_threshold": compute_test_metrics(
            y_test_bin,
            y_pred_test,
            proba_test,
            s_test,
        ),
        "metrics_threshold_optimizer": compute_test_metrics(
            y_test_bin,
            y_pred_to,
            proba_test,
            s_test,
        ),
    }


def main() -> None:
    df = load_german_credit()
    print("Base approval rate by Sex:")
    print(base_approval_rate_by_group(df).to_string(index=False))

    X, y, s = make_xy_s(df)
    X_train, X_test, y_train, y_test, s_train, s_test = (
        stratified_train_test_split(X, y, s)
    )

    result = train_fair_credit_model(
        X_train,
        X_test,
        y_train,
        y_test,
        s_train,
        s_test,
    )
    metrics = result["metrics_threshold"]
    save_metrics(metrics, OUTPUT_METRICS_PATH)

    print(f"\nFrozen threshold (OOF/train): {result['threshold']:.4f}")
    print("OOF selection metrics:", result["oof_metrics"])
    print("Test metrics (frozen threshold):", metrics)
    print(f"Wrote {OUTPUT_METRICS_PATH.resolve()}")


if __name__ == "__main__":
    main()
