import subprocess
import time
import csv
import os
import json
import argparse
import ast
from datetime import datetime

RESULTS_CSV = "benchmark_results.csv"

def count_lloc(file_path):
    if not os.path.exists(file_path):
        return 0
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            tree = ast.parse(f.read(), filename=file_path)
        return sum(1 for node in ast.walk(tree) if isinstance(node, (ast.stmt, ast.expr)))
    except Exception:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith("#")]
        return len(lines)

def init_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Enfoque", "Script_Name",
                "Pass_k_Pytest", "Test_Failure_Reason", "Defect_Density_KLOC", "Sonar_Reliability_Rating", "Code_Complexity_Sonar",
                "TTR_Seconds", "TTF_Seconds", "Num_Interactions", "LLOC",
                "McCabe_Cyclomatic_Complexity", "Code_Duplication_Pct",
                "Open_Vulnerabilities_Count", "Security_Hotspots_Density",
                "F1_Score", "Precision", "Recall", "ROC_AUC", "Disparate_Impact", "Demographic_Parity_Diff", "Pass_Fairness_Rule",
                "NASA_TLX_Score", "Technical_Debt_Minutes"
            ])

def run_benchmark(enfoque, script_path, ttf=0, num_interactions=1, nasa_tlx=0, sonarqube_json=None):
    init_csv()
    print("\n==================================================")
    print(f" Executing Benchmark - Enfoque: {enfoque}")
    print(f" Script: {script_path}")
    print("==================================================\n")

    lloc = count_lloc(script_path)

    start_time = time.time()
    exec_error = ""
    try:
        exec_process = subprocess.run(
            ["python", script_path], 
            capture_output=True, 
            text=True, 
            timeout=180
        )
        if exec_process.returncode != 0:
            exec_error = f"Runtime Error: {exec_process.stderr.strip()[:100]}"
    except subprocess.TimeoutExpired:
        exec_error = "Timeout Error (exceeded 180s)"

    ttr_seconds = round(time.time() - start_time, 2)

    f1, precision, recall, roc_auc, di, dpd = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if os.path.exists("output_metrics.json"):
        try:
            with open("output_metrics.json", "r", encoding="utf-8", errors="ignore") as f:
                m = json.load(f)
                f1 = m.get("f1_score", 0.0)
                precision = m.get("precision", 0.0)
                recall = m.get("recall", 0.0)
                roc_auc = m.get("roc_auc", 0.0)
                di = m.get("disparate_impact", 0.0)
                dpd = m.get("demographic_parity_difference", 0.0)
        except Exception as e:
            print(f"Error reading output_metrics.json: {e}")

    defect_density, sonar_reliability, sonar_complexity = 0.0, "A", 0
    mccabe_complexity, duplication_pct = 0, 0.0
    vulnerabilities, security_hotspots, tech_debt_min = 0, 0, 0

    if sonarqube_json and os.path.exists(sonarqube_json):
        try:
            with open(sonarqube_json, "r", encoding="utf-8", errors="ignore") as f:
                sq = json.load(f)
                defect_density = sq.get("defect_density", 0.0)
                sonar_reliability = sq.get("reliability_rating", "A")
                sonar_complexity = sq.get("cyclomatic_complexity", 0)
                mccabe_complexity = sq.get("mccabe_complexity", sonar_complexity)
                duplication_pct = sq.get("duplication_pct", 0.0)
                vulnerabilities = sq.get("vulnerabilities", 0)
                security_hotspots = sq.get("security_hotspots", 0)
                tech_debt_min = sq.get("technical_debt_minutes", 0)
        except Exception as e:
            print(f"Error reading SonarQube JSON: {e}")

    test_process = subprocess.run(["pytest", "tests/"], capture_output=True, text=True)
    pytest_passed = (test_process.returncode == 0)
    pass_fairness = (di >= 0.80 and f1 >= 0.70)

    failure_reasons = []
    if exec_error:
        failure_reasons.append(exec_error)
    if not os.path.exists("output_metrics.json"):
        failure_reasons.append("Missing output_metrics.json")
    else:
        if di < 0.80:
            failure_reasons.append(f"Fairness Violation (DI={di:.2f} < 0.80)")
        if f1 < 0.70:
            failure_reasons.append(f"Low Performance (F1={f1:.2f} < 0.70)")
    
    failure_reason_str = "PASSED ALL TESTS" if (pytest_passed and not failure_reasons) else " | ".join(failure_reasons)

    with open(RESULTS_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            enfoque,
            os.path.basename(script_path),
            pytest_passed, failure_reason_str, defect_density, sonar_reliability, sonar_complexity,
            ttr_seconds, ttf, num_interactions, lloc,
            mccabe_complexity, duplication_pct,
            vulnerabilities, security_hotspots,
            f1, precision, recall, roc_auc, di, dpd, pass_fairness,
            nasa_tlx, tech_debt_min
        ])

    print("\n--------------------------------------------------")
    print(f" RECORDED IN {RESULTS_CSV}:")
    print(f" - Pass@k (PyTest):         {pytest_passed}")
    print(f" - Test Status/Reason:      {failure_reason_str}")
    print(f" - F1-Score:                {f1:.4f} | Disparate Impact: {di:.4f}")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner Completo del Benchmark (6 Dimensiones)")
    parser.add_argument("--enfoque", required=True, choices=["IA_Pura", "Correccion_Humana", "HITL_Iterativo"])
    parser.add_argument("--script", required=True, help="Ruta al script de la solucion")
    parser.add_argument("--ttf", type=float, default=0.0, help="Time to First Draft en segundos")
    parser.add_argument("--interactions", type=int, default=1, help="Numero de interacciones/prompts")
    parser.add_argument("--nasa_tlx", type=float, default=0.0, help="Puntuacion NASA-TLX (0-100)")
    parser.add_argument("--sonar_json", type=str, default=None, help="Archivo JSON con metricas de SonarQube")

    args = parser.parse_args()
    run_benchmark(
        args.enfoque, args.script, 
        ttf=args.ttf, 
        num_interactions=args.interactions, 
        nasa_tlx=args.nasa_tlx, 
        sonarqube_json=args.sonar_json
    )