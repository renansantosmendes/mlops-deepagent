"""Tools de treinamento: análise do histórico (MLflow) para guiar o treino atual
e AutoML (FLAML) para seleção de modelo + hiperparâmetros."""

import json
import logging
import time

import joblib
import numpy as np

from .state import artifact_path, load_state, save_state

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "mlops-deepagent"


def analyze_training_history(max_runs: int = 30) -> str:
    """Analisa runs anteriores no MLflow para guiar o treinamento atual.

    Retorna: melhor métrica histórica (baseline a bater), famílias de modelos
    que mais venceram (para priorizar no AutoML) e tempo médio de treino
    (para calibrar o budget). Use ANTES de run_automl_training.

    Args:
        max_runs: Quantidade máxima de runs históricos a considerar.
    """
    import mlflow

    logger.info("📊 Analisando histórico de treinamentos no MLflow...")
    
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        logger.info("🆕 Nenhum histórico encontrado (primeiro ciclo)")
        return (
            "Nenhum histórico encontrado (primeiro ciclo). Recomendação: usar AutoML com "
            "estimadores padrão e budget de ~60s; não há baseline a bater."
        )
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="params.best_estimator != ''",  # só runs consolidados do pipeline
        order_by=["attributes.start_time DESC"],
        max_results=max_runs,
    )
    if not runs:
        return "Experimento existe mas sem runs. Trate como primeiro ciclo."

    rows = []
    for r in runs:
        rows.append(
            {
                "run_id": r.info.run_id,
                "best_estimator": r.data.params.get("best_estimator"),
                "primary_metric": r.data.metrics.get("primary_metric"),
                "train_seconds": r.data.metrics.get("train_seconds"),
            }
        )
    scored = [r for r in rows if r["primary_metric"] is not None]
    if not scored:
        return "Histórico sem métricas registradas. Trate como primeiro ciclo."

    best = max(scored, key=lambda r: r["primary_metric"])
    winners: dict[str, int] = {}
    for r in scored:
        if r["best_estimator"]:
            winners[r["best_estimator"]] = winners.get(r["best_estimator"], 0) + 1
    avg_time = float(np.mean([r["train_seconds"] for r in scored if r["train_seconds"]] or [60]))

    guidance = {
        "n_previous_runs": len(scored),
        "best_metric_so_far": round(best["primary_metric"], 5),
        "best_run_id": best["run_id"],
        "winning_estimators": winners,
        "avg_train_seconds": round(avg_time, 1),
        "recommendation": (
            f"Priorizar estimadores {sorted(winners, key=winners.get, reverse=True)[:3]} "
            f"e usar budget >= {max(60, int(avg_time))}s. Baseline a bater: "
            f"{best['primary_metric']:.5f}."
        ),
    }
    save_state({"history_guidance": guidance})
    return json.dumps(guidance, indent=2, ensure_ascii=False)


def run_automl_training(time_budget_seconds: int = 60, estimator_list: str = "") -> str:
    """Roda AutoML (FLAML) nos dados processados: testa LightGBM, XGBoost, RandomForest,
    ExtraTrees e regressão logística/linear, escolhendo modelo + hiperparâmetros.
    Loga tudo no MLflow e salva o melhor modelo em disco.

    Args:
        time_budget_seconds: Budget de tempo do AutoML (use a sugestão do histórico).
        estimator_list: Estimadores a priorizar, separados por vírgula
            (ex.: 'lgbm,xgboost,rf'). Vazio = todos os padrão do FLAML.
    """
    import mlflow
    from flaml import AutoML

    logger.info("🤖 Iniciando treinamento AutoML (FLAML)...")
    
    state = load_state()
    if "preprocessor_path" not in state:
        logger.error("❌ Erro: dados ainda não processados")
        return "ERRO: dados ainda não processados. Rode process_data primeiro."

    X_train = np.load(artifact_path("X_train.npy"))
    y_train = np.load(artifact_path("y_train.npy"), allow_pickle=True)
    task = state.get("task_type", "classification")
    metric = "roc_auc" if task == "classification" and len(np.unique(y_train)) == 2 else (
        "macro_f1" if task == "classification" else "r2"
    )
    
    logger.info(f"📊 Task: {task} | Métrica: {metric}")
    logger.info(f"⏱️  Budget de tempo: {time_budget_seconds}s")
    logger.info(f"📈 Amostras de treino: {len(X_train)}")

    automl = AutoML()
    settings = {
        "task": task,
        "metric": metric,
        "time_budget": time_budget_seconds,
        "seed": 42,
        "verbose": 0,
        "mlflow_logging": False,  # evita 1 run por trial; logamos só o run consolidado
    }
    estimators = [e.strip() for e in estimator_list.split(",") if e.strip()]
    # Normalize estimator names to FLAML-supported aliases
    if estimators:
        mapping = {
            # common synonyms
            "extras_trees": "extra_tree",
            "extra_trees": "extra_tree",
            "random_forest": "rf",
            "xgb": "xgboost",
        }
        normalized = []
        for est in estimators:
            key = est.strip().lower()
            if key in mapping:
                normalized.append(mapping[key])
            else:
                normalized.append(key)
        estimators = normalized
        settings["estimator_list"] = estimators
        logger.info(f"🎯 Estimadores priorizados (normalizados): {estimators}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    t0 = time.time()
    logger.info("🔄 Executando AutoML...")
    with mlflow.start_run() as run:
        automl.fit(X_train=X_train, y_train=y_train, **settings)
        logger.info(f"✅ AutoML concluído em {time.time() - t0:.1f}s")
        elapsed = time.time() - t0
        # FLAML retorna 1 - métrica como loss para métricas de maximização
        primary = 1 - automl.best_loss
        logger.info(f"🏆 Melhor modelo: {automl.best_estimator}")
        logger.info(f"📊 Métrica ({metric}): {primary:.4f}")
        mlflow.log_params(
            {
                "best_estimator": automl.best_estimator,
                "task": task,
                "metric": metric,
                "time_budget": time_budget_seconds,
                **{f"hp_{k}": v for k, v in (automl.best_config or {}).items()},
            }
        )
        mlflow.log_metrics(
            {"primary_metric": float(primary), "train_seconds": float(elapsed)}
        )
        run_id = run.info.run_id

    model_path = artifact_path("model.joblib")
    joblib.dump(automl.model.estimator, model_path)
    save_state(
        {
            "model_path": model_path,
            "best_estimator": automl.best_estimator,
            "best_config": automl.best_config,
            "cv_metric_name": metric,
            "cv_metric_value": float(primary),
            "mlflow_run_id": run_id,
        }
    )

    hist = state.get("history_guidance", {})
    baseline = hist.get("best_metric_so_far")
    comparison = ""
    if baseline is not None:
        comparison = (
            f" Baseline histórico: {baseline:.5f} -> "
            + ("SUPEROU o histórico." if primary > baseline else "NÃO superou o histórico.")
        )
    return (
        f"AutoML concluído em {elapsed:.0f}s. Melhor modelo: {automl.best_estimator} "
        f"({metric} CV = {primary:.5f}). Config: {automl.best_config}. "
        f"MLflow run: {run_id}.{comparison}"
    )
