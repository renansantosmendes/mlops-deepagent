"""Tool de avaliação: métricas no conjunto de teste + quality gate de aprovação."""

import json

import joblib
import numpy as np

from .state import artifact_path, load_state, save_state


def evaluate_model(min_metric_to_approve: float = 0.0) -> str:
    """Avalia o modelo treinado no conjunto de teste (holdout) e aplica um quality gate.

    Classificação: accuracy, precision/recall/F1 macro e ROC-AUC (binário).
    Regressão: R2, MAE, RMSE. Loga as métricas no run do MLflow e salva
    evaluation_report.json com o veredito approved=True/False.

    Args:
        min_metric_to_approve: Threshold mínimo da métrica principal
            (ROC-AUC/F1 para classificação, R2 para regressão) para aprovar o modelo.
            0.0 = aprova sempre.
    """
    import mlflow
    from sklearn import metrics as skm

    state = load_state()
    if "model_path" not in state:
        return "ERRO: nenhum modelo treinado. Rode run_automl_training primeiro."

    model = joblib.load(state["model_path"])
    X_test = np.load(artifact_path("X_test.npy"))
    y_test = np.load(artifact_path("y_test.npy"), allow_pickle=True)
    task = state.get("task_type", "classification")

    y_pred = model.predict(X_test)
    report: dict = {"task": task}

    if task == "classification":
        report["accuracy"] = float(skm.accuracy_score(y_test, y_pred))
        report["precision_macro"] = float(skm.precision_score(y_test, y_pred, average="macro"))
        report["recall_macro"] = float(skm.recall_score(y_test, y_pred, average="macro"))
        report["f1_macro"] = float(skm.f1_score(y_test, y_pred, average="macro"))
        primary = report["f1_macro"]
        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            report["roc_auc"] = float(skm.roc_auc_score(y_test, proba))
            primary = report["roc_auc"]
        report["confusion_matrix"] = skm.confusion_matrix(y_test, y_pred).tolist()
    else:
        report["r2"] = float(skm.r2_score(y_test, y_pred))
        report["mae"] = float(skm.mean_absolute_error(y_test, y_pred))
        report["rmse"] = float(np.sqrt(skm.mean_squared_error(y_test, y_pred)))
        primary = report["r2"]

    report["primary_test_metric"] = primary
    report["approved"] = bool(primary >= min_metric_to_approve)
    report["threshold"] = min_metric_to_approve

    run_id = state.get("mlflow_run_id")
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in report.items() if isinstance(v, (int, float))}
            )

    path = artifact_path("evaluation_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    save_state({"evaluation_report_path": path, "model_approved": report["approved"]})

    verdict = (
        "APROVADO para registro/deploy"
        if report["approved"]
        else f"REPROVADO (métrica {primary:.4f} < threshold {min_metric_to_approve})"
    )
    return f"Avaliação: {verdict}.\n{json.dumps(report, indent=2)}"
