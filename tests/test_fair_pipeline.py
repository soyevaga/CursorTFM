import pytest
import os
import json

METRICS_FILE = "output_metrics.json"

def test_metrics_file_exists():
    """Valida si la solución generó el reporte estandarizado de métricas."""
    assert os.path.exists(METRICS_FILE), f"Error: No se encontró el archivo '{METRICS_FILE}'."

def test_fairness_threshold():
    """Valida que se cumpla la restricción de equidad (Regla del 80%: Disparate Impact >= 0.80)."""
    assert os.path.exists(METRICS_FILE), "El archivo de métricas no existe."
    with open(METRICS_FILE, "r") as f:
        data = json.load(f)
    
    di = data.get("disparate_impact", 0.0)
    assert di >= 0.80, f"Violación de Equidad: Disparate Impact es {di:.2f} (se requiere >= 0.80)."

def test_performance_threshold():
    """Valida que el rendimiento predictivo mínimo no se haya degradado excesivamente."""
    assert os.path.exists(METRICS_FILE), "El archivo de métricas no existe."
    with open(METRICS_FILE, "r") as f:
        data = json.load(f)
    
    f1 = data.get("f1_score", 0.0)
    assert f1 >= 0.70, f"Rendimiento bajo: F1-Score es {f1:.2f} (se requiere >= 0.70)."
