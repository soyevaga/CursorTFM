"""
Pipeline de clasificación de riesgo crediticio sobre German Credit Data.

Mejoras respecto a la versión base:
- Clasificador más robusto (HistGradientBoosting) con búsqueda de hiperparámetros
  orientada a F1 y umbral de decisión ajustado con predicciones out-of-fold.
- Mitigación de sesgo sobre Sex: umbrales por grupo bajo paridad demográfica
  (y ThresholdOptimizer de Fairlearn como alternativa), eligiendo la estrategia
  en validación interna, nunca en test.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
TEST_SIZE = 0.20
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output_metrics.json"
DI_LOW, DI_HIGH = 0.80, 1.25

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)
UCI_COLUMNS = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "class",
]


def load_german_credit() -> pd.DataFrame:
    try:
        bunch = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
        frame = bunch.frame.copy()
        print(f"Dataset cargado desde OpenML: {frame.shape[0]} filas, {frame.shape[1]} columnas.")
        return frame
    except Exception as exc:
        print(f"OpenML no disponible ({exc}). Descargando desde UCI...")
        return _load_from_uci()


def _load_from_uci() -> pd.DataFrame:
    with urllib.request.urlopen(UCI_URL, timeout=60) as response:
        raw = response.read().decode("utf-8")
    rows = [line.split() for line in raw.strip().splitlines() if line.strip()]
    frame = pd.DataFrame(rows, columns=UCI_COLUMNS)
    numeric_cols = [
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
        "class",
    ]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["class"] = frame["class"].map({1: "good", 2: "bad"})
    print(f"Dataset cargado desde UCI: {frame.shape[0]} filas, {frame.shape[1]} columnas.")
    return frame


def extract_sex(personal_status: pd.Series) -> pd.Series:
    text = personal_status.astype(str).str.lower()
    sex = np.where(text.str.contains("female"), "female", "male")
    return pd.Series(sex, index=personal_status.index, name="sex")


def engineer_features(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    if {"credit_amount", "duration"}.issubset(out.columns):
        duration = pd.to_numeric(out["duration"], errors="coerce").clip(lower=1)
        amount = pd.to_numeric(out["credit_amount"], errors="coerce")
        out["credit_per_month"] = amount / duration
        out["log_credit_amount"] = np.log1p(amount.clip(lower=0))
    if "age" in out.columns:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
        out["young_applicant"] = (out["age"] < 25).astype(int)
    return out


def prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    data = frame.copy()
    data.columns = [str(c).strip() for c in data.columns]

    target_col = "class" if "class" in data.columns else data.columns[-1]
    y_raw = data[target_col].astype(str).str.lower().str.strip()
    y = y_raw.map({"bad": 1, "good": 0, "2": 1, "1": 0})
    if y.isna().any():
        raise ValueError(f"Valores de clase no reconocidos: {sorted(y_raw.unique())}")

    sensitive_col = next(
        (c for c in ("personal_status", "personal_status_sex", "status_sex") if c in data.columns),
        None,
    )
    if sensitive_col is None:
        raise ValueError("No se encontró la columna de estado personal / sexo.")

    sex = extract_sex(data[sensitive_col])
    x = engineer_features(data.drop(columns=[target_col]))
    return x, y.astype(int), sex


def build_pipeline(x: pd.DataFrame) -> Pipeline:
    numeric_features = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in x.columns if c not in numeric_features]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])


def tune_classifier(pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> Pipeline:
    param_distributions = {
        "clf__learning_rate": np.array([0.03, 0.05, 0.08, 0.12, 0.2]),
        "clf__max_depth": [3, 4, 5, 6, None],
        "clf__max_iter": [150, 250, 400, 600],
        "clf__min_samples_leaf": [8, 15, 25, 40],
        "clf__l2_regularization": [0.0, 0.1, 0.5, 1.0, 5.0],
        "clf__class_weight": ["balanced"],
    }
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="f1",
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x, y)
    print(f"Mejor F1 en CV: {search.best_score_:.4f} | params={search.best_params_}")
    return search.best_estimator_


def oof_probabilities(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    oof = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, va) in enumerate(cv.split(x, y), start=1):
        fold_model = clone(model)
        fold_model.fit(x.iloc[tr], y.iloc[tr])
        oof[va] = fold_model.predict_proba(x.iloc[va])[:, 1]
        print(f"  OOF fold {fold}/5 listo")
    return oof


def apply_group_thresholds(
    proba: np.ndarray, sex: pd.Series, thr_female: float, thr_male: float
) -> np.ndarray:
    groups = sex.astype(str).str.lower().to_numpy()
    thresholds = np.where(groups == "female", thr_female, thr_male)
    return (proba >= thresholds).astype(int)


def disparate_impact_ratio(y_pred: np.ndarray, sensitive: pd.Series) -> float:
    pred = np.asarray(y_pred)
    groups = sensitive.astype(str).str.lower().to_numpy()
    unprivileged = pred[groups == "female"]
    privileged = pred[groups == "male"]
    rate_u = float(np.mean(unprivileged)) if len(unprivileged) else 0.0
    rate_p = float(np.mean(privileged)) if len(privileged) else 0.0
    if rate_p == 0.0:
        return 1.0 if rate_u == 0.0 else float("inf")
    return rate_u / rate_p


def fairness_ok(di: float, dpd: float) -> bool:
    return np.isfinite(di) and DI_LOW <= di <= DI_HIGH and dpd <= 0.10


def candidate_metrics(y_true: pd.Series, y_pred: np.ndarray, sex: pd.Series) -> dict:
    di = disparate_impact_ratio(y_pred, sex)
    dpd = float(
        demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sex)
    )
    return {
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "di": float(di),
        "dpd": abs(dpd),
        "dpd_signed": dpd,
    }


def score_candidate(stats: dict) -> tuple:
    """Prioriza F1 alto y, a igualdad, cumplimiento de equidad."""
    return (
        int(fairness_ok(stats["di"], stats["dpd"])),
        stats["f1"],
        -stats["dpd"],
        -abs(np.log(stats["di"]) if stats["di"] > 0 and np.isfinite(stats["di"]) else 10.0),
    )


def search_group_thresholds(proba: np.ndarray, y: pd.Series, sex: pd.Series) -> dict:
    grid = np.linspace(0.20, 0.80, 31)
    best_any = None
    best_fair = None
    y_arr = y.to_numpy()

    for thr_f in grid:
        for thr_m in grid:
            pred = apply_group_thresholds(proba, sex, thr_f, thr_m)
            stats = candidate_metrics(y, pred, sex)
            stats.update({"thr_female": float(thr_f), "thr_male": float(thr_m), "kind": "group"})
            key = score_candidate(stats)
            if best_any is None or key > score_candidate(best_any):
                best_any = stats
            if fairness_ok(stats["di"], stats["dpd"]):
                if best_fair is None or stats["f1"] > best_fair["f1"]:
                    best_fair = stats

    # Umbral global (referencia de rendimiento)
    best_global = None
    for thr in grid:
        pred = (proba >= thr).astype(int)
        stats = candidate_metrics(y, pred, sex)
        stats.update({"thr_female": float(thr), "thr_male": float(thr), "kind": "global"})
        if best_global is None or stats["f1"] > best_global["f1"]:
            best_global = stats

    chosen = best_fair if best_fair is not None else best_any
    print(
        f"Umbral global OOF F1={best_global['f1']:.4f} | "
        f"elegido {chosen['kind']} F1={chosen['f1']:.4f} "
        f"DI={chosen['di']:.3f} DPD={chosen['dpd']:.3f} "
        f"(f={chosen['thr_female']:.2f}, m={chosen['thr_male']:.2f})"
    )
    del y_arr
    return chosen


def fit_threshold_optimizer(
    estimator: Pipeline, x: pd.DataFrame, y: pd.Series, sex: pd.Series
) -> ThresholdOptimizer:
    mitigator = ThresholdOptimizer(
        estimator=estimator,
        constraints="demographic_parity",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    mitigator.fit(x, y, sensitive_features=sex)
    return mitigator


def main() -> None:
    frame = load_german_credit()
    x, y, sex = prepare_xy(frame)

    x_train, x_test, y_train, y_test, sex_train, sex_test = train_test_split(
        x, y, sex, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Ajustando HistGradientBoosting (RandomizedSearchCV, score=F1)...")
    model = tune_classifier(build_pipeline(x_train), x_train, y_train)

    print("Calculando probabilidades out-of-fold para umbral y equidad...")
    oof = oof_probabilities(model, x_train, y_train)
    chosen = search_group_thresholds(oof, y_train, sex_train)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(x_train, y_train, clf__sample_weight=sample_weight)
    test_proba = model.predict_proba(x_test)[:, 1]

    pred_group = apply_group_thresholds(
        test_proba, sex_test, chosen["thr_female"], chosen["thr_male"]
    )

    print("Ajustando ThresholdOptimizer (paridad demográfica)...")
    optimizer = fit_threshold_optimizer(model, x_train, y_train, sex_train)
    pred_to = optimizer.predict(x_test, sensitive_features=sex_test, random_state=RANDOM_STATE)

    oof_group_stats = chosen
    oof_to = optimizer.predict(x_train, sensitive_features=sex_train, random_state=RANDOM_STATE)
    oof_to_stats = candidate_metrics(y_train, oof_to, sex_train)
    oof_to_stats["kind"] = "threshold_optimizer"

    use_optimizer = score_candidate(oof_to_stats) > score_candidate(oof_group_stats)
    y_pred = pred_to if use_optimizer else pred_group
    strategy = "ThresholdOptimizer" if use_optimizer else "umbrales por grupo"
    print(f"Estrategia de equidad seleccionada (OOF): {strategy}")

    metrics = {
        "f1_score": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "disparate_impact": float(disparate_impact_ratio(y_pred, sex_test)),
        "demographic_parity_difference": float(
            demographic_parity_difference(
                y_true=y_test, y_pred=y_pred, sensitive_features=sex_test
            )
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nMetricas guardadas en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
