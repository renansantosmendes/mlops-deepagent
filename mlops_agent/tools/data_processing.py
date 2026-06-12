"""Tool de processamento: limpeza, pipeline sklearn (imputação, encoding, scaling)
e split treino/teste. O preprocessor é salvo para ser empacotado junto do modelo."""

import joblib
import numpy as np
import pandas as pd

from .state import artifact_path, load_state, save_state


def process_data(test_size: float = 0.2, drop_columns: str = "") -> str:
    """Limpa e processa o dataset bruto e gera os splits de treino/teste.

    Remove duplicatas e colunas constantes, imputa nulos (mediana/moda),
    aplica one-hot em categóricas e StandardScaler em numéricas.
    Salva X_train/X_test/y_train/y_test e o preprocessor (joblib).

    Args:
        test_size: Fração para teste (padrão 0.2).
        drop_columns: Colunas extras a remover, separadas por vírgula (ex.: IDs).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    state = load_state()
    if "raw_data_path" not in state:
        return "ERRO: nenhum dado coletado ainda."
    df = pd.read_parquet(state["raw_data_path"])
    target = state["target_column"]

    # Limpeza básica
    df = df.drop_duplicates()
    to_drop = [c.strip() for c in drop_columns.split(",") if c.strip() and c.strip() in df.columns]
    const_cols = [c for c in df.columns if c != target and df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=to_drop + const_cols)
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    task = state.get("task_type", "classification")
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    Xtr = preprocessor.fit_transform(X_train)
    Xte = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())
    np.save(artifact_path("X_train.npy"), Xtr)
    np.save(artifact_path("X_test.npy"), Xte)
    np.save(artifact_path("y_train.npy"), y_train.to_numpy())
    np.save(artifact_path("y_test.npy"), y_test.to_numpy())
    prep_path = artifact_path("preprocessor.joblib")
    joblib.dump(preprocessor, prep_path)

    save_state(
        {
            "preprocessor_path": prep_path,
            "feature_names": feature_names,
            "raw_feature_columns": list(X.columns),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "dropped_columns": to_drop + const_cols,
            "train_rows": int(Xtr.shape[0]),
            "test_rows": int(Xte.shape[0]),
        }
    )
    return (
        f"Processamento concluído: {Xtr.shape[0]} linhas de treino, {Xte.shape[0]} de teste, "
        f"{Xtr.shape[1]} features após encoding. Removidas: {to_drop + const_cols or 'nenhuma'}. "
        f"Preprocessor salvo em {prep_path}."
    )
