"""
Pipeline de clasificación de riesgo crediticio sobre German Credit Data.

Carga el dataset (OpenML, con respaldo UCI), preprocesa variables numéricas y
categóricas, entrena un modelo y evalúa rendimiento y equidad. Los resultados
se guardan en output_metrics.json.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output_metrics.json"

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
    """Carga German Credit Data desde OpenML o, si falla, desde UCI."""
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
    # UCI: 1 = good, 2 = bad
    frame["class"] = frame["class"].map({1: "good", 2: "bad"})
    print(f"Dataset cargado desde UCI: {frame.shape[0]} filas, {frame.shape[1]} columnas.")
    return frame


def extract_sex(personal_status: pd.Series) -> pd.Series:
    """Deriva el atributo sensible sexo a partir de personal_status."""
    text = personal_status.astype(str).str.lower()
    sex = np.where(text.str.contains("female"), "female", "male")
    return pd.Series(sex, index=personal_status.index, name="sex")


def prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separa características, etiqueta de riesgo y atributo sensible."""
    data = frame.copy()
    data.columns = [str(c).strip() for c in data.columns]

    target_col = "class" if "class" in data.columns else data.columns[-1]
    y_raw = data[target_col].astype(str).str.lower().str.strip()
    # 1 = cliente de riesgo (crédito malo), 0 = crédito bueno
    y = y_raw.map({"bad": 1, "good": 0, "2": 1, "1": 0})
    if y.isna().any():
        raise ValueError(f"Valores de clase no reconocidos: {sorted(y_raw.unique())}")

    sensitive_col = None
    for candidate in ("personal_status", "personal_status_sex", "status_sex"):
        if candidate in data.columns:
            sensitive_col = candidate
            break
    if sensitive_col is None:
        raise ValueError("No se encontró la columna de estado personal / sexo.")

    sex = extract_sex(data[sensitive_col])
    x = data.drop(columns=[target_col])
    return x, y.astype(int), sex


def build_pipeline(x: pd.DataFrame) -> Pipeline:
    numeric_features = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in x.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", model)])


def disparate_impact_ratio(y_pred: np.ndarray, sensitive: pd.Series) -> float:
    """
    Impacto dispar: tasa de positivos (riesgo) en el grupo no privilegiado
    (mujeres) dividida por la del grupo privilegiado (hombres).
    """
    pred = np.asarray(y_pred)
    groups = sensitive.astype(str).str.lower().to_numpy()
    unprivileged = pred[groups == "female"]
    privileged = pred[groups == "male"]
    rate_u = float(np.mean(unprivileged)) if len(unprivileged) else 0.0
    rate_p = float(np.mean(privileged)) if len(privileged) else 0.0
    if rate_p == 0.0:
        return 1.0 if rate_u == 0.0 else float("inf")
    return rate_u / rate_p


def main() -> None:
    frame = load_german_credit()
    x, y, sex = prepare_xy(frame)

    x_train, x_test, y_train, y_test, sex_train, sex_test = train_test_split(
        x,
        y,
        sex,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    del sex_train  # el atributo sensible solo se usa en evaluación

    pipeline = build_pipeline(x_train)
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "f1_score": float(f1_score(y_test, y_pred, pos_label=1)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "disparate_impact": float(disparate_impact_ratio(y_pred, sex_test)),
        "demographic_parity_difference": float(
            demographic_parity_difference(
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=sex_test,
            )
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nMétricas guardadas en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
