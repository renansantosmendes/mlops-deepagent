"""Deep agent de MLOps construído com a lib `deepagents` (LangChain).

Orquestra o ciclo completo de uma solução de ML:
coleta -> análise (qualidade/drift) -> processamento -> treino (AutoML guiado
pelo histórico) -> avaliação -> registro (MLflow) -> deploy (FastAPI).

Uso:
    python -m mlops_agent.agent "Rode o ciclo completo com o dataset demo"
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Adiciona o diretório raiz ao path se executado diretamente
if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv
from deepagents import create_deep_agent

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from mlops_agent.tools.data_analysis import analyze_data_quality, detect_data_drift
from mlops_agent.tools.data_collection import (
    collect_data_from_file,
    collect_data_from_sql,
    collect_data_from_url,
    generate_demo_dataset,
)
from mlops_agent.tools.data_processing import process_data
from mlops_agent.tools.deployment import check_api_health, deploy_model_api, stop_api
from mlops_agent.tools.evaluation import evaluate_model
from mlops_agent.tools.registry import register_model
from mlops_agent.tools.training import analyze_training_history, run_automl_training

SYSTEM_PROMPT = """Você é um engenheiro de MLOps autônomo responsável pelo ciclo de vida
completo de soluções de Machine Learning.

Fluxo padrão (use write_todos para planejar antes de executar):
1. COLETA: use a tool de coleta adequada à fonte informada pelo usuário
   (arquivo, URL, SQL ou dataset demo).
2. ANÁLISE: rode analyze_data_quality e detect_data_drift. Se houver drift ou
   issues graves, reporte e decida (ou pergunte) antes de seguir.
3. PROCESSAMENTO: rode process_data, removendo colunas-ID apontadas na análise
   de qualidade via drop_columns.
4. TREINO: SEMPRE rode analyze_training_history primeiro e use a recomendação
   (estimadores vencedores e budget) ao chamar run_automl_training.
5. AVALIAÇÃO: rode evaluate_model com um threshold sensato
   (ex.: 0.7 de ROC-AUC/F1 para classificação, 0.5 de R2 para regressão),
   a menos que o usuário defina outro. Se reprovado, volte ao passo 4 com mais
   budget ou ao passo 3 com outro processamento — no máximo 2 retentativas.
6. REGISTRO: se aprovado, rode register_model.
7. DEPLOY: rode deploy_model_api e valide com check_api_health.

Regras:
- Nunca registre nem faça deploy de modelo reprovado na avaliação.
- Sempre compare o resultado atual com o baseline histórico quando existir.
- Ao final, escreva um resumo executivo: dados, qualidade, modelo escolhido,
  métricas, versão registrada e URL da API.
- Delegue etapas pesadas aos subagentes quando disponível, mantendo o contexto limpo.
"""

DATA_SUBAGENT = {
    "name": "data-engineer",
    "description": (
        "Especialista em dados: coleta, análise de qualidade, detecção de drift e "
        "processamento/feature engineering. Delegue as etapas 1-3 do pipeline a ele."
    ),
    "system_prompt": (
        "Você é um engenheiro de dados. Colete os dados da fonte indicada, rode a análise "
        "de qualidade e de drift, e processe os dados (removendo colunas-ID/problemáticas). "
        "Reporte um resumo objetivo: linhas/colunas, issues encontrados, drift e o que foi "
        "feito no processamento."
    ),
    "tools": [
        collect_data_from_file,
        collect_data_from_url,
        collect_data_from_sql,
        generate_demo_dataset,
        analyze_data_quality,
        detect_data_drift,
        process_data,
    ],
}

TRAINING_SUBAGENT = {
    "name": "ml-trainer",
    "description": (
        "Especialista em treinamento: analisa o histórico de runs no MLflow, roda AutoML "
        "(FLAML) e avalia o modelo no holdout. Delegue as etapas 4-5 a ele."
    ),
    "system_prompt": (
        "Você é um cientista de ML. SEMPRE analise o histórico de treinamentos antes de "
        "treinar e use as recomendações (estimadores e budget). Rode o AutoML, avalie no "
        "teste com quality gate e reporte métricas + comparação com o baseline histórico. "
        "Se reprovar, tente no máximo mais 2 vezes com budget maior."
    ),
    "tools": [analyze_training_history, run_automl_training, evaluate_model],
}

DEPLOY_SUBAGENT = {
    "name": "deploy-engineer",
    "description": (
        "Especialista em release: registra o modelo aprovado no MLflow Registry e faz o "
        "deploy na API FastAPI, validando com smoke test. Delegue as etapas 6-7 a ele."
    ),
    "system_prompt": (
        "Você é um engenheiro de release. Registre o modelo aprovado no MLflow Model "
        "Registry, suba a API FastAPI e valide com check_api_health. Nunca prossiga se o "
        "modelo não estiver aprovado. Reporte versão registrada e URL da API."
    ),
    "tools": [register_model, deploy_model_api, check_api_health, stop_api],
}

ALL_TOOLS = (
    DATA_SUBAGENT["tools"] + TRAINING_SUBAGENT["tools"] + DEPLOY_SUBAGENT["tools"]
)


def build_agent(model: str = "openai:gpt-5-nano"):
    """Cria o deep agent orquestrador com os subagentes do ciclo de ML."""
    logger.info("🤖 Inicializando MLOps Deep Agent...")
    logger.info(f"📦 Modelo: {model}")
    logger.info(f"🔧 Ferramentas disponíveis: {len(ALL_TOOLS)}")
    logger.info(f"👥 Subagentes: {len([DATA_SUBAGENT, TRAINING_SUBAGENT, DEPLOY_SUBAGENT])} (Data, Training, Deploy)")
    
    agent = create_deep_agent(
        model=model,
        tools=ALL_TOOLS,  # orquestrador também pode agir direto, se preferir
        system_prompt=SYSTEM_PROMPT,
        subagents=[DATA_SUBAGENT, TRAINING_SUBAGENT, DEPLOY_SUBAGENT],
    )
    
    logger.info("✅ Agente inicializado com sucesso!\n")
    return agent


def main() -> None:
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info("🚀 MLOps Deep Agent - Iniciando execução")
    logger.info("="*80)
    
    task = (
        " ".join(sys.argv[1:])
        or "Rode o ciclo completo de ML usando o dataset demo de classificação."
    )
    
    logger.info(f"📋 Tarefa solicitada: {task}")
    logger.info("="*80 + "\n")
    
    agent = build_agent()
    
    logger.info("▶️  Executando agente...\n")
    logger.info("-"*80)
    
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": 100},
        )
        
        logger.info("-"*80)
        logger.info("✅ Execução concluída com sucesso!\n")
        logger.info("="*80)
        logger.info("📊 RESULTADO FINAL")
        logger.info("="*80)
        print(result["messages"][-1].content)
        
        elapsed_time = datetime.now() - start_time
        logger.info("\n" + "="*80)
        logger.info(f"⏱️  Tempo total de execução: {elapsed_time}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error("-"*80)
        logger.error(f"❌ Erro durante a execução: {str(e)}")
        logger.error("-"*80)
        raise


if __name__ == "__main__":
    main()
