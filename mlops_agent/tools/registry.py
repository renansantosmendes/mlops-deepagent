"""Tool de registro: empacota preprocessor + modelo num pipeline sklearn e
registra no MLflow Model Registry com versionamento e alias."""

import joblib

from .state import artifact_path, load_state, save_state


def register_model(model_name: str = "mlops-deepagent-model", alias: str = "champion") -> str:
    """Registra o modelo aprovado no MLflow Model Registry.

    Empacota o preprocessor + estimador em um único sklearn Pipeline (recebe dados
    brutos na predição), cria uma nova versão no registry e aplica o alias.

    Args:
        model_name: Nome do modelo no registry.
        alias: Alias da versão (ex.: 'champion', 'staging').
    """
    import mlflow
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline

    state = load_state()
    if not state.get("model_approved"):
        return "ERRO: o modelo não foi aprovado na avaliação (ou não foi avaliado). Não registrarei."

    preprocessor = joblib.load(state["preprocessor_path"])
    model = joblib.load(state["model_path"])
    full_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Exemplo de input com os dados brutos para assinatura do modelo
    raw = pd.read_parquet(state["raw_data_path"])
    input_example = raw.drop(columns=[state["target_column"]]).head(3)
    # Garante dtypes float64 nas numéricas para a assinatura
    for c in input_example.select_dtypes(include=np.number).columns:
        input_example[c] = input_example[c].astype("float64")

    mlflow.set_experiment("mlops-deepagent")
    with mlflow.start_run(run_id=state.get("mlflow_run_id")):
        info = mlflow.sklearn.log_model(
            full_pipeline,
            name="model_pipeline",
            registered_model_name=model_name,
            input_example=input_example,
        )

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(model_name, alias, str(latest))

    # Salva o pipeline completo em disco para o deploy local com FastAPI
    pipeline_path = artifact_path("full_pipeline.joblib")
    joblib.dump(full_pipeline, pipeline_path)

    save_state(
        {
            "registered_model_name": model_name,
            "registered_model_version": latest,
            "registered_model_alias": alias,
            "model_uri": info.model_uri,
            "full_pipeline_path": pipeline_path,
        }
    )
    return (
        f"Modelo registrado: '{model_name}' versão {latest} (alias '{alias}'). "
        f"URI: {info.model_uri}. Pipeline completo salvo em {pipeline_path}."
    )
