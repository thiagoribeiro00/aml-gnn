🚀 Projeto: DeepGraph-AML: Detecção de Redes de Lavagem com Graph Neural Networks (GNN) no Neo4j AuraDB
🎯 Objetivo
Construir um pipeline que utiliza o Neo4j AuraDB como fonte única da verdade para treinar uma rede Graph Convolutional Network (GCN) ou GraphSAGE, capaz de identificar nós suspeitos (contas/lavadores de dinheiro) baseando-se não apenas nas características da transação, mas na topologia da rede (quem se conecta com quem).
💡 Por que este projeto impressiona na entrevista?
Alinhamento Técnico: Usa exatamente a stack mencionada (Neo4j, Python, DL).
Diferencial "Staff": Mostra que você entende que em AML, o relacionamento (grafo) é mais importante que o atributo isolado.
Escalabilidade: Demonstra como extrair subgrafos massivos do AuraDB para treinamento eficiente.
🏗️ Arquitetura da Solução
1. Camada de Dados (Neo4j AuraDB)
Schema:
(:Account {id, risk_score, kyc_status})
(:Transaction {id, amount, timestamp, currency})
(:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->(:Account)
Feature Engineering no Banco (Cypher):
Cálculo de métricas de centralidade (PageRank, Betweenness) diretamente no Neo4j usando a Graph Data Science (GDS) Library.
Essas métricas viram features estáticas dos nós.
2. Camada de Extração (Python Connector)
Script Python que se conecta ao AuraDB.
Extrai subgrafos relevantes (ex: vizinhança de 2 hops de contas suspeitas) convertendo para tensores compatíveis com PyTorch Geometric (PyG).
Desafio técnico: Lidar com a extração eficiente de grandes volumes sem estourar a memória.
3. Camada de Modelo (Deep Learning - PyTorch Geometric)
Modelo: Implementação de GraphSAGE (indutivo, ideal para grafos dinâmicos como transações financeiras) ou GAT (Graph Attention Network, para explicar quais vizinhos influenciaram a decisão).
Tarefa: Node Classification (Binária: Legítimo vs. Ilícito).
Loss Function: Weighted Cross-Entropy (para lidar com o desbalanceamento extremo de dados de fraude).
4. Camada de Inferência & Feedback
O modelo prediz a probabilidade de fraude para novos nós.
Os resultados são escritos de volta no Neo4j (SET n.predicted_risk = ...) para visualização imediata no Bloom ou Dashboard.
🛠️ Stack Tecnológico
Database: Neo4j AuraDB (Free ou Professional).
Linguagem: Python 3.9+.
Libs de Grafo: neo4j, graphdatascience (GDS library client).
Deep Learning: pytorch, torch-geometric (PyG).
Data Manipulation: pandas, numpy.
Ambiente: Google Colab (com GPU ativada para treino).



wise-aml-graph-project/
│
├── .env                          # Variáveis de ambiente (URI Neo4j, Senhas)
├── requirements.txt              # Dependências exatas
├── README.md                     # Documentação do projeto (Crucial para Staff)
│
├── data/                         # Dados brutos e processados (gitignored)
│   ├── raw/                      # Dataset Elliptic original (.csv)
│   ├── interim/                  # Dados limpos intermediários
│   └── processed/                # Grafos projetados ou tensores salvos
│
├── src/                          # Código Fonte Principal
│   │
│   ├── __init__.py
│   │
│   ├── domain/                   # CAMADA DE DOMÍNIO (Puro Python, sem deps externas)
│   │   ├── entities.py           # Classes: TransactionNode, AccountEdge
│   │   ├── value_objects.py      # Objetos de valor: RiskScore, TransactionID
│   │   └── repositories.py       # Interfaces (Abstract Base Classes) dos repositórios
│   │                             # Ex: class GraphRepository(ABC): def get_subgraph(...)
│   │
│   ├── use_cases/                # CAMADA DE CASOS DE USO (Regras de Negócio)
│   │   ├── data_ingestion.py     # Lógica de carregar e validar dados
│   │   ├── feature_engineering.py# Cálculo de centralidades (lógica, não implementação DB)
│   │   ├── train_model.py        # Orquestração do treino (loop, validação)
│   │   └── predict_and_save.py   # Fluxo de inferência e escrita de resultados
│   │
│   ├── adapters/                 # CAMADA DE ADAPTADORES (Tradução de formatos)
│   │   ├── neo4j_adapter.py      # Implementa GraphRepository usando neo4j.Driver
│   │   ├── pytorch_adapter.py    # Converte dados do DB para torch_geometric.Data
│   │   └── logger_adapter.py     # Configuração de logs estruturados
│   │
│   ├── infrastructure/           # CAMADA DE INFRAESTRUTURA (Detalhes externos)
│   │   ├── config.py             # Leitura de .env e configurações globais
│   │   ├── database.py           # Singleton do driver Neo4j
│   │   └── storage.py            # Leitura/Escrita em disco ou S3
│   │
│   └── models/                   # Definição dos Modelos de DL
│       ├── gnn_architecture.py   # Classes PyTorch (GraphSAGE, GAT)
│       └── loss_functions.py     # Funções de perda customizadas (Weighted CE)
│
├── tests/                        # Testes Unitários e de Integração
│   ├── test_domain.py
│   ├── test_use_cases.py
│   └── test_adapters.py          # Mocks do Neo4j
│
├── notebooks/                    # Apenas para EDA exploratório e prototipagem rápida
│   ├── 01_eda_elliptic.ipynb
│   └── 02_prototype_gnn.ipynb
│
└── scripts/                      # Scripts executáveis para CLI
    ├── run_training.py
    └── deploy_predictions.py


adicione uma pasta de deploy, que irá conter os scripts de deploy do modelo, 

screscente um makefile para rodar usando docker, automatizar o deploy usando o gcp.

adicione um arquivo .gitignore para ignorar os arquivos que não devem ser versionados.

adicione o .env com todas env necessarias.

