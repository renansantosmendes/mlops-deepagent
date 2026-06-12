"""Tools de análise de dados: qualidade (nulos, duplicatas, outliers, cardinalidade)
e drift (teste KS para numéricas, PSI e qui-quadrado para categóricas)."""

import json

import numpy as np
import pandas as pd

from .state import artifact_path, load_state, save_state


def analyze_data_quality() -> str:
    """Gera um relatório de qualidade do dataset bruto: nulos, duplicatas, outliers (IQR),
    constantes, cardinalidade e balanceamento do alvo. Salva quality_report.json."""
    state = load_state()
    if "raw_data_path" not in state:
        return "ERRO: nenhum dado coletado ainda. Use uma tool de coleta primeiro."
    df = pd.read_parquet(state["raw_data_path"])
    target = state["target_column"]

    report: dict = {"n_rows": len(df), "n_cols": df.shape[1], "issues": []}

    nulls = df.isna().mean()
    report["null_pct_by_column"] = {c: round(float(v), 4) for c, v in nulls.items() if v > 0}
    for c, v in nulls.items():
        if v > 0.3:
            report["issues"].append(f"Coluna '{c}' com {v:.0%} de nulos (>30%)")

    dup = int(df.duplicated().sum())
    report["duplicated_rows"] = dup
    if dup > 0:
        report["issues"].append(f"{dup} linhas duplicadas")

    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    report["constant_columns"] = const_cols
    if const_cols:
        report["issues"].append(f"Colunas constantes: {const_cols}")

    num_cols = df.select_dtypes(include=np.number).columns.drop(target, errors="ignore")
    outliers = {}
    for c in num_cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)
        pct = float(mask.mean())
        if pct > 0.01:
            outliers[c] = round(pct, 4)
    report["outlier_pct_by_column"] = outliers

    cat_cols = df.select_dtypes(exclude=np.number).columns
    high_card = [c for c in cat_cols if df[c].nunique() > 0.5 * len(df)]
    if high_card:
        report["issues"].append(f"Alta cardinalidade (possível ID): {high_card}")

    if df[target].nunique() <= 20:
        report["task_type"] = "classification"
        dist = df[target].value_counts(normalize=True)
        report["target_distribution"] = {str(k): round(float(v), 4) for k, v in dist.items()}
        if dist.min() < 0.10:
            report["issues"].append(f"Alvo desbalanceado (classe minoritária {dist.min():.1%})")
    else:
        report["task_type"] = "regression"
        report["target_stats"] = {
            "mean": float(df[target].mean()),
            "std": float(df[target].std()),
        }

    path = artifact_path("quality_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    save_state({"quality_report_path": path, "task_type": report["task_type"]})

    verdict = "OK para prosseguir" if len(report["issues"]) == 0 else "Atenção aos issues"
    return f"Relatório de qualidade salvo em {path}. {verdict}.\n{json.dumps(report, indent=2)[:2500]}"


def detect_data_drift(reference_path: str = "") -> str:
    """Detecta drift entre o dataset atual e um dataset de referência.

    Usa teste Kolmogorov-Smirnov + PSI para colunas numéricas e
    qui-quadrado + PSI para categóricas. Salva drift_report.json.

    Args:
        reference_path: Caminho do dataset de referência (csv/parquet). Se vazio,
            usa o baseline salvo de um treinamento anterior, se existir.
    """
    from scipy import stats

    state = load_state()
    if "raw_data_path" not in state:
        return "ERRO: nenhum dado coletado ainda."
    current = pd.read_parquet(state["raw_data_path"])

    if not reference_path:
        reference_path = state.get("baseline_data_path", "")
    if not reference_path:
        # Sem referência: salva o dataset atual como baseline para os próximos ciclos.
        baseline = artifact_path("baseline.parquet")
        current.to_parquet(baseline)
        save_state({"baseline_data_path": baseline})
        return (
            "Nenhum baseline de referência encontrado. O dataset atual foi salvo como "
            "baseline para detecção de drift em ciclos futuros. Prossiga com o pipeline."
        )

    ref = (
        pd.read_parquet(reference_path)
        if reference_path.endswith(".parquet")
        else pd.read_csv(reference_path)
    )

    results = {}
    drifted = []
    common = [c for c in current.columns if c in ref.columns]
    for c in common:
        cur, re_ = current[c].dropna(), ref[c].dropna()
        if pd.api.types.is_numeric_dtype(current[c]):
            ks_stat, p = stats.ks_2samp(re_, cur)
            psi = _psi_numeric(re_, cur)
            has_drift = bool(p < 0.05 and psi > 0.2)
            results[c] = {
                "type": "numeric",
                "ks_pvalue": round(float(p), 6),
                "psi": round(psi, 4),
                "drift": has_drift,
            }
        else:
            psi = _psi_categorical(re_, cur)
            has_drift = bool(psi > 0.2)
            results[c] = {"type": "categorical", "psi": round(psi, 4), "drift": has_drift}
        if has_drift:
            drifted.append(c)

    report = {
        "n_columns_checked": len(common),
        "drifted_columns": drifted,
        "drift_share": round(len(drifted) / max(len(common), 1), 3),
        "details": results,
    }
    path = artifact_path("drift_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    save_state({"drift_report_path": path, "drift_detected": len(drifted) > 0})

    if drifted:
        return (
            f"DRIFT DETECTADO em {len(drifted)}/{len(common)} colunas: {drifted}. "
            f"Recomenda-se retreinar o modelo. Relatório: {path}"
        )
    return f"Sem drift significativo ({len(common)} colunas verificadas). Relatório: {path}"


def _psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    r, _ = np.histogram(ref, bins=edges)
    c, _ = np.histogram(cur, bins=edges)
    return _psi_from_counts(r, c)


def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    r = ref.value_counts().reindex(cats, fill_value=0).to_numpy()
    c = cur.value_counts().reindex(cats, fill_value=0).to_numpy()
    return _psi_from_counts(r, c)


def _psi_from_counts(r: np.ndarray, c: np.ndarray) -> float:
    rp = np.clip(r / max(r.sum(), 1), 1e-6, None)
    cp = np.clip(c / max(c.sum(), 1), 1e-6, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))
