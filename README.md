# MLOps Deep Agent

Deep agent (lib [`deepagents`](https://github.com/langchain-ai/deepagents) da LangChain) que gerencia e automatiza **todo o ciclo de vida de uma solução de ML** por meio de tools:

| Etapa | Tools | Tecnologia |
|---|---|---|
| 1. Coleta | `collect_data_from_file` / `_url` / `_sql`, `generate_demo_dataset` | pandas, SQLAlchemy |
| 2. Análise | `analyze_data_quality`, `detect_data_drift` | KS-test, PSI, qui-quadrado |
| 3. Processamento | `process_data` | sklearn Pipeline (imputação, OHE, scaling) + split |
| 4. Treino | `analyze_training_history` → `run_automl_training` | **MLflow** (histórico guia o treino) + **FLAML** (AutoML) |
| 5. Avaliação | `evaluate_model` | métricas no holdout + quality gate |
| 6. Registro | `register_model` | MLflow Model Registry (versão + alias `champion`) |
| 7. Deploy | `deploy_model_api`, `check_api_health`, `stop_api` | **FastAPI** + uvicorn |

## Arquitetura

```
Orquestrador (create_deep_agent)
├── write_todos / filesystem (built-in do deepagents)
├── subagente data-engineer   → coleta, qualidade, drift, processamento
├── subagente ml-trainer      → histórico, AutoML, avaliação
└── subagente deploy-engineer → registro, deploy FastAPI, smoke test
```

As tools se comunicam por um estado compartilhado em `pipeline_artifacts/state.json`
(datasets, preprocessor, modelo, relatórios), então o agente pode retomar de qualquer etapa.

**Pontos de inteligência do fluxo:**
- `analyze_training_history` lê os runs anteriores no MLflow e devolve baseline a bater,
  estimadores que mais venceram e budget sugerido — o agente usa isso para parametrizar o AutoML.
- `detect_data_drift` salva o 1º dataset como baseline; em ciclos futuros sinaliza drift e
  recomenda retreino.
- `evaluate_model` aplica quality gate; `register_model` **recusa** modelos reprovados.
- A API serve o sklearn Pipeline completo (preprocessor + modelo), aceitando dados brutos.

## Instalação

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

## Uso

```bash
# Ciclo completo com dataset demo
python -m mlops_agent.agent "Rode o ciclo completo de ML com o dataset demo de classificação"

# Com dados reais
python -m mlops_agent.agent "Colete data/churn.csv com alvo 'churn', analise qualidade e drift, \
processe, treine com AutoML guiado pelo histórico, avalie com threshold 0.75, registre e faça deploy"

# Só monitoramento de drift de um ciclo novo
python -m mlops_agent.agent "Colete data/churn_junho.csv (alvo 'churn') e verifique drift; \
se houver drift, retreine e promova só se superar o baseline"
```

A API sobe em `http://localhost:8000`:

```bash
curl localhost:8000/health
curl localhost:8000/model-info
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"records":[{"f0":0.1,"f1":-1.2, "...": 0}]}'
```

UI do MLflow (experimentos + registry): `mlflow ui`

## Estrutura

```
mlops_agent/
  agent.py              # deep agent + subagentes
  tools/
    state.py            # estado compartilhado do pipeline
    data_collection.py  # etapa 1
    data_analysis.py    # etapa 2 (qualidade + drift)
    data_processing.py  # etapa 3
    training.py         # etapa 4 (histórico + AutoML)
    evaluation.py       # etapa 5 (quality gate)
    registry.py         # etapa 6 (MLflow Registry)
    deployment.py       # etapa 7 (FastAPI)
serving/app.py          # API FastAPI servida no deploy
requirements.txt
```

## Notas

- Modelo do agente: configurável em `build_agent()` (padrão `anthropic:claude-sonnet-4-5`);
  qualquer chat model LangChain funciona.
- Para produção, troque o tracking do MLflow para um servidor remoto
  (`MLFLOW_TRACKING_URI`) e o deploy local por um container (o `serving/app.py`
  já é o entrypoint pronto para um Dockerfile com uvicorn).
- Requer `deepagents>=0.4` (specs de subagente com `system_prompt`).
