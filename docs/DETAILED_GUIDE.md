# 🏦 AML-GNN: Guia Técnico de Deep Learning e Detecção de Fraude em Grafos

Este guia fornece uma visão profunda sobre a arquitetura, as decisões de engenharia e os algoritmos de Redes Neurais de Grafos (GNN) utilizados no projeto de Monitoramento AML.

---

## 1. Visão Geral e Descoberta de Dados (Data Discovery)

### O Dataset: Elliptic Data Set
O projeto utiliza o dataset da **Elliptic**, focado em transações de Bitcoin. O diferencial é que as transações não são independentes; elas formam um grafo direcionado.
- **Nós (Transactions):** 203.769 nós.
- **Arestas (Fund Flows):** 234.355 conexões.
- **Classes:** 
  1. **Ilícito (fraudulento):** ~2% (O "alvo" do nosso modelo).
  2. **Lícito (legítimo):** ~21%.
  3. **Desconhecido:** ~77% (Nós não rotulados usados para aprendizado estrutural).

### Características das Features (165 total)
As variáveis foram divididas em duas categorias principais pela Elliptic:
1.  **Features Locais (1-94):** Dados brutos da transação (valor, taxas, timestep, número de inputs/outputs).
2.  **Features Agregadas (95-165):** Estatísticas dos vizinhos imediatos (média de valor dos vizinhos, desvio padrão das taxas dos vizinhos, etc.).

> **Insight:** Mesmo com as features agregadas, um modelo de ML comum (XGBoost/RF) perde a **hierarquia** e a **conectividade profunda**. O GNN consegue "olhar" além dos vizinhos imediatos e capturar saltos distantes na rede.

---

## 2. Engenharia e Tratamento de Dados

### Padronização (Normalization)
Como as transações de Bitcoin variam de centavos a milhões, os dados são altamente enviesados.
- **Técnica:** Utilizamos o `RobustScaler` (Scikit-Learn).
- **Por que:** Ao contrário do StandardScaler, o RobustScaler utiliza o intervalo interquartil (IQR), sendo imune a outliers extremos que são comuns em dados de fraude.

### Divisão Temporal (Temporal Split)
Em AML, o passado não prevê o futuro perfeitamente (**Concept Drift**). 
- **Estratégia:** Os dados são divididos em 49 timesteps. 
- **Train:** Primeiros 34 timesteps.
- **Val:** Timesteps 35 a 40.
- **Test:** Últimos 9 timesteps.
Isso garante que o modelo seja testado em um "futuro" que ele nunca viu, simulando um cenário real de produção.

---

## 3. Arquiteturas de Deep Learning (GNN)

Escolhemos duas arquiteturas de ponta para este desafio:

### A. GraphSAGE (SAGEConv)
**Por que escolher:** Foi projetado para grafos gigantes onde novos nós aparecem constantemente.
- **Intuicão:** Em vez de aprender um ID de nó (como no Node2Vec), ele aprende uma **função de agregação**. 
- **Como funciona:** Para cada nó $v$, ele amostra vizinhos $u \in N(v)$, concatena as features de $v$ com a média das features de $u$, e passa por uma camada densa com ReLU. 
- **Vantagem:** É muito estável e menos propenso ao "over-smoothing" (quando todos os nós ficam com a mesma representação).

### B. GAT (Graph Attention Network)
**Por que escolher:** Nem todo vizinho é igual. Uma transação vinda de um "Mixer" de cripto deve ter mais peso que uma vinda de uma Exchange regulada.
- **Intuição:** O GAT calcula coeficientes de atenção $a_{ij}$ entre o nó $i$ e o vizinho $j$. 
- **Mecanismo:** Através de *Multi-head Attention*, o modelo "olha" para diferentes aspectos da vizinhança simultaneamente (ex: um head foca no valor, outro na frequência temporal).
- **Vantagem:** Maior poder expressivo para detectar sub-redes criminosas muito camufladas.

---

## 4. Estratégia de Treinamento e Otimização

### Combatendo o Desbalanceamento (Mask-based Weighted Loss)
Com apenas 2% de casos positivos, um modelo preguiçoso diria que "toda transação é lícita" e teria 98% de acurácia, mas 0% de utilidade.
- **Nossa solução:** Ajustamos a `CrossEntropyLoss` com o parâmetro `weight`. 
- **Cálculo:** O peso da classe ilícita é inversamente proporcional à sua frequência. Errar uma fraude "custou" muito mais caro para o modelo durante o Backpropagation do que errar uma transação legítima.

### Otimização do Threshold (Limiar)
A saída do modelo é uma probabilidade entre 0 e 1. O padrão é 0.5, mas em AML, o custo de um falso negativo (deixar um criminoso passar) é diferente do falso positivo.
- **Processo:** Após o treinamento, rodamos um loop no conjunto de Validação testando limiares de 0.05 a 0.95.
- **Meta:** Encontrar o ponto onde o **F1-Score** é máximo.

---

## 5. Métricas de Validação: Além da Acurácia

Para este projeto, ignoramos a Acurácia e focamos em:
1.  **Precision (Precisão):** Das que o modelo marcou como fraude, quantas eram realmente? (Evita travar contas de bons clientes).
2.  **Recall (Revocação):** Das fraudes que existiam no total, quantas o modelo conseguiu pegar? (Métrica de segurança).
3.  **F1-Score:** A média harmônica entre as duas. É a nossa métrica principal para o "Model Selection".

---

## 6. GraphRAG: O Elo com a Linguagem Natural

O componente final integra o **Neo4j** com o **Gemini 1.5 Pro**.
- **O Desafio:** GNNs operam em espaços vetoriais (embeddings) que humanos não entendem.
- **A Solução:** Usamos o GNN para encontrar o culpado e o **GraphRAG** para contar a história.
- **Fluxo:** 
    1. O GNN sinaliza a transação `TX_99`.
    2. O `GraphRetriever` busca no Neo4j as 10 conexões mais fortes da `TX_99`.
    3. O `Gemini` recebe o dump do grafo e gera um relatório: *"O modelo considerou esta transação de alto risco (0.87) porque ela recebeu fundos de 2 endereços na lista de sanções e repassou o valor em menos de 10 minutos (Layering)."*

---

## 7. Tech Stack Resumido
- **Deep Learning:** PyTorch Geometric (PyG).
- **Banco de Grafos:** Neo4j (via Cypher queries).
- **Orquestração LLM:** LangChain & LangGraph.
- **Observabilidade:** LangSmith (RAG Tracing).
- **Interface:** Streamlit (Instant response via Lazy Loading).
