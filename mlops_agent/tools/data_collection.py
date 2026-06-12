"""Tools de coleta de dados: CSV/Parquet local, URL, SQL e dataset sintético de exemplo."""

import logging
import os
import pandas as pd
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from .state import artifact_path, save_state
from .environment import get_env_var

logger = logging.getLogger(__name__)


def collect_data_from_file(path: str, target_column: str) -> str:
    """Coleta dados de um arquivo local CSV ou Parquet e o registra como dataset bruto do pipeline.

    Args:
        path: Caminho do arquivo .csv ou .parquet.
        target_column: Nome da coluna alvo (label) para o problema de ML.
    """
    logger.info(f"📁 Coletando dados do arquivo: {path}")
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    logger.info(f"✅ Arquivo carregado: {len(df)} linhas, {len(df.columns)} colunas")
    return _register_raw(df, target_column, source=f"file:{path}")


def collect_data_from_url(url: str, target_column: str) -> str:
    """Coleta dados de uma URL que aponte para um CSV e o registra como dataset bruto.

    Args:
        url: URL pública de um arquivo CSV.
        target_column: Nome da coluna alvo (label).
    """
    # Expand environment variable placeholders like ${VAR} using helper
    if isinstance(url, str) and url.startswith("${") and url.endswith("}"):
        var_name = url[2:-1]
        expanded = get_env_var(var_name, None)
        if not expanded:
            logger.error(
                f"❌ Não foi possível expandir a URL de dados a partir do placeholder '{url}'."
            )
            raise FileNotFoundError(
                f"URL de dados não encontrada para {var_name} (esperado em variável de ambiente)."
            )
        url = expanded
    df = pd.read_csv(url)
    return _register_raw(df, target_column, source=f"url:{url}")


def collect_data_from_sql(connection_string: str, query: str, target_column: str) -> str:
    """Coleta dados executando uma query SQL (via SQLAlchemy) e registra o resultado como dataset bruto.

    Args:
        connection_string: String de conexão SQLAlchemy (ex.: postgresql://user:pwd@host/db).
        query: Query SQL de extração.
        target_column: Nome da coluna alvo (label).
    """
    from sqlalchemy import create_engine

    engine = create_engine(connection_string)
    df = pd.read_sql(query, engine)
    return _register_raw(df, target_column, source="sql")


def generate_demo_dataset(n_samples: int = 2000, task: str = "classification") -> str:
    """Gera um dataset sintético (sklearn) para demonstrar o pipeline ponta a ponta.

    Args:
        n_samples: Número de linhas.
        task: 'classification' ou 'regression'.
    """
    from sklearn.datasets import make_classification, make_regression

    logger.info(f"🎲 Gerando dataset sintético: {task}")
    logger.info(f"📈 Amostras: {n_samples}, Features: 12")
    
    if task == "classification":
        X, y = make_classification(
            n_samples=n_samples, n_features=12, n_informative=6, random_state=42
        )
    else:
        X, y = make_regression(n_samples=n_samples, n_features=12, noise=0.2, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    
    logger.info(f"✅ Dataset gerado com sucesso!")
    return _register_raw(df, "target", source=f"synthetic:{task}")


def _register_raw(df: pd.DataFrame, target_column: str, source: str) -> str:
    if target_column not in df.columns:
        logger.error(f"❌ Erro: coluna alvo '{target_column}' não existe")
        return (
            f"ERRO: coluna alvo '{target_column}' não existe. "
            f"Colunas disponíveis: {list(df.columns)}"
        )
    raw_path = artifact_path("raw.parquet")
    logger.info(f"💾 Salvando dados brutos em: {raw_path}")
    df.to_parquet(raw_path)
    save_state(
        {
            "raw_data_path": raw_path,
            "target_column": target_column,
            "data_source": source,
            "n_rows": len(df),
            "n_cols": df.shape[1],
        }
    )
    return (
        f"Dataset coletado de {source}: {len(df)} linhas x {df.shape[1]} colunas. "
        f"Salvo em {raw_path}. Alvo: '{target_column}'. "
        f"Colunas: {list(df.columns)[:30]}"
    )
