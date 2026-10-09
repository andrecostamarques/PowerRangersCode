# Diário de Bordo do Projeto (Journal Log)

> **Documento de Registro Histórico de Atividades e Decisões dos Agentes**  
> **Localização:** `journal-docs/journal.md`  
> **Objetivo:** Manter a memória contínua das tarefas executadas, scripts desenvolvidos, resultados obtidos e decisões de projeto.

---

## 📌 Registro de Entradas (Log de Tasks)

### 🗓️ Entry #001 — Engenharia e Concepção da Máscara de Consenso (Consensus Mask)
* **Data:** Julho / 2026
* **Status:** Concluído e Validado

#### 🎯 Motivação e Diagnóstico do Problema:
No treinamento com **Máscaras Dinâmicas Aprendíveis** (`SelectionMask` individual), cada modelo tentava aprender sua própria seleção espacial de pixels. No entanto, foram observados três problemas fundamentais:
1. **Viés Arquitetural e Variação Estocástica:** Cada rede neural (LeNet vs ResNet vs CNN simples) convergia para máscaras com densidades drasticamente diferentes (de 0.19% a 32.70% de pixels ativos).
2. **Colapso por Super-Pruning (ResNet34):** A ResNet34 forçou a esparsidade a um nível extremo de **0.19% de pixels ativos** (apenas ~370 pixels), resultando no colapso do modelo e queda da acurácia de **85.65% para 70.74%** (-14.91%).
3. **Falta de Generalização Espacial:** Uma máscara aprendida por um modelo específico não refletia o consenso geral sobre onde está a informação física relevante do objeto (galáxias).

#### 💡 A Solução: Máscara Consenso (Inteligência Coletiva)
Criar uma **máscara espacial estática unificada** agregando o aprendizado das 4 arquiteturas para filtrar o ruído de fundo e reter apenas as características fundamentais onde há concordância entre os modelos.

#### ⚙️ Detalhamento Algorítmico (`src/scripts/generate_consensus_mask.py`):
1. **Binarização Individual:** Para cada um dos 4 modelos (`LeNet256`, `ResNet20`, `ResNet34`, `SimpleCNNRGB`), extrai-se a matriz de pesos da máscara no melhor checkpoint (`checkpoint_epoch_X.pt` onde $X > 1$) e calcula-se a máscara binária individual:
   $$M_i = \text{round}(\sigma(W_i)) \in \{0, 1\}^{C \times H \times W}$$
2. **Média Element-wise (Agregação):**
   $$\bar{M} = \frac{1}{N} \sum_{i=1}^N M_i \quad (N = 4)$$
3. **Votação Majoritária (Limiarização $\ge 0.5$):**
   $$M_{\text{consenso}} = \mathbb{I}(\bar{M} \ge 0.5)$$
   *Pixels ativados em pelo menos 2 dos 4 modelos são mantidos.*
4. **Re-codificação de Logits para PyTorch:** Para garantir a perfeita compatibilidade com o módulo `SelectionMask` durante o carregamento, os pesos contínuos da máscara consensual foram definidos via logits extremos:
   $$W_{\text{consenso}} = \begin{cases} +10.0 & \text{se } M_{\text{consenso}} = 1.0 \quad (\sigma(10.0) \approx 1.0) \\ -10.0 & \text{se } M_{\text{consenso}} = 0.0 \quad (\sigma(-10.0) \approx 0.0) \end{cases}$$
5. **Serialização:** Salvamento do checkpoint autocontido em `checkpoints/consensus_mask.pt` contendo `mask_model_obj` e `mask_state_dict`.

#### 📊 Resultados Impactantes do Consenso:
* **Taxa de Atividade:** A máscara de consenso estabilizou em **10.11% de pixels ativos** (descarte eficaz de **89.89%** da imagem).
* **Resgate da ResNet34:** A acurácia da ResNet34 saltou de **70.74%** (na máscara dinâmica) para **85.59%** (no consenso), praticamente igualando a baseline de 100% de pixels (85.65%) e **superando o F1-Score da baseline** (**84.43%** vs **84.06%**).
* **Herança de Features no SimpleCNN:** O `SimpleCNNRGB` aumentou sua acurácia de **74.75%** para **77.88%** (+3.13%), aproveitando o mapa espacial refinado por redes mais profundas.

---

### 🗓️ Entry #002 — Análise Comparativa, Extração de Métricas e Gerador de Gráficos
* **Data:** Julho / 2026
* **Status:** Concluído

#### 🎯 Objetivos e Ações:
1. **Extração de Métricas (`src/scripts/extract_metrics.py`):** Script para ler os checkpoints e gerar tabelas Markdown com Acurácia, Precision, Recall, Macro F1-Score e Esparsidade.
2. **Visualização de Máscaras (`src/scripts/plot_all_masks.py`):** Geração da imagem comparativa 3x4 em RGB das máscaras aprendidas por modelo e regime.
3. **Gráfico Comparativo de Barras (`plot_comparison_chart.py`):** Desenvolvimento do script que gera a imagem `plot_comparativo_metricas.png` (Subplots lado a lado de Acurácia e F1-Score por arquitetura).
4. **Análise de Variabilidade (`analise_comparativa_insights.md`):** Documentação da natureza estocástica do treino STE vs. a **robustez estrutural da compressão de ~90%**.
5. **Auditoria da Apresentação 2.0:** Revisão do PDF e correção de erros de digitação em tabelas (ex: correção do delta da ResNet34 no consenso de `-6%` para `-0.06%`).

---

### 🗓️ Entry #004 — Preparação da Expansão para Food-101 e Otimizações de Hardware
* **Data:** Setembro / 2026
* **Status:** Concluído e Em Execução de Treino

#### 🎯 Objetivos e Ações Executadas:
1. **Suporte Multi-Dataset no Código:**
   * Adicionada a classe `Food101` ao `DatasetsDict.py` com redimensionamento para $256 \times 256$ e normalização RGB ImageNet.
   * Tornou-se dinâmico o parâmetro `num_classes` nos modelos (`LeNet256`, `SimpleCNNRGB`, `ResNet20`, `ResNet34`) e scripts (`train_classifier.py`, `run_experiments.py`, `generate_consensus_mask.py`), suportando 101 classes no Food-101 e 10 no CIFAR-10/Galaxy10.
2. **Otimização Extrema de Hardware (`TrainingConfig.py`):**
   * Configurado `num_workers = 8`, `persistent_workers = True` e `pin_memory = True` para máximo aproveitamento do AMD Ryzen 5 3600 (12 threads) e NVIDIA RTX 5060 Ti (32GB RAM).
3. **Automação de Scripts de Treino:**
   * Criados os scripts `run_all_food101.sh` e `run_all_cifar10.sh`.
   * Criados os documentos de controle `temp-todo.md`, `next-task-datasets.md` e `prompt.md`.

### 🗓️ Entry #005 — Visualizador de Logs, Geração da Consensus Mask do Food-101 e Padronização de Execução
* **Data:** Outubro / 2026
* **Status:** Concluído e Em Execução de Retreino com Máscara Consenso

#### 🎯 Objetivos e Ações Executadas:
1. **Script de Visualização `plot_training_log.py` (`src/scripts/plot_training_log.py`):**
   * Desenvolvido script para ler diretamente do arquivo `training_log.csv` sem depender de recarregar checkpoints `.pt`.
   * Exibe graficamente o tradeoff entre Perdas (Total, Model, Mask), Acurácia de Validação e o parâmetro $\lambda$. Inclui eixo secundário para visualizar % de pixels ativos e destaque vertical da época selecionada.
2. **Ajuste e Otimização do Gerador de Consenso (`generate_consensus_mask.py`):**
   * Atualizada a resolução de caminhos para suportar a estrutura hierárquica de subpastas por dataset (`checkpoints/food101/`).
   * Implementada ordenação estrita por número de época para selecionar a época mais recente (`epoch 200`).
   * Otimizada a instanciação do `SelectionMask` em memória a partir da forma dos tensores, evitando I/O redundante de 4.3 GB da época 1.
   * Gerada e salva a **Máscara Consenso do Food-101** em `checkpoints/consensus_mask_food101.pt` e `checkpoints/food101/consensus_mask_food101.pt`.
3. **Padronização Estrita do Protocolo de Execução:**
   * Registrada no `prompt.md` e `baseline-powerrangers.md` a diretriz obrigatória de navegar até o diretório `src/scripts/` (`cd src/scripts`) antes de executar qualquer script do projeto.
4. **Padronização Automática dos Nomes de Diretórios (`train_classifier.py`):**
   * Atualizada a resolução de `training_id`. Quando o argumento `--mask_checkpoint` contiver a palavra `consensus`, o script nomeia automaticamente os diretórios de saída como `food101_<modelo>_consensus_mask` (ex: `checkpoints/food101/food101_lenet256_consensus_mask/`), garantindo conformidade com a convenção estabelecida no Galaxy10.

---

### 🗓️ Entry #006 — Extração de Métricas Oficiais e Consolidação Científica do Dataset Food-101
* **Data:** Outubro / 2026
* **Status:** Concluído e Registrado

#### 🎯 Objetivos e Ações Executadas:
1. **Refatoração do Extrator de Métricas (`src/scripts/extract_metrics.py`):**
   * Adicionado argumento de linha de comando `--dataset` (suportando `food101` e `galaxy10`).
   * Implementada busca dinâmica de checkpoints tanto no diretório raiz `checkpoints/` quanto nas subpastas específicas (`checkpoints/food101/`).
   * Adicionada seleção ordenada da melhor/última época para extrair acurácia, precisão, recall, F1-Score e % de pixels ativos.
2. **Consolidação das Métricas Oficiais (101 Classes):**
   * **Taxa de Atividade da Consensus Mask:** A máscara de consenso do Food-101 estabilizou em **58.04% de pixels ativos** (compressão de **41.96%**).
   * **Resgate do Colapso do `SimpleCNNRGB`:** Na máscara dinâmica, o `SimpleCNNRGB` zerou a retenção de pixels (0.00%), resultando em colapso completo da rede (**0.74%** de acurácia). A **Consensus Mask** resgatou o modelo para **14.72%** (+13.98% de ganho).
   * **Ganho na ResNet20:** A `ResNet20` subiu de **48.70%** (na máscara dinâmica com 26.15% de pixels) para **52.98%** no consenso (+4.28%).
   * **ResNet34 e LeNet256:** Mantiveram performance sólida com atenuação suave da perda de resolução espacial.

---

## 💡 Decisões Técnicas e Detalhes de Implementação Cruciais

### 1. Mecanismo Straight-Through Estimator (STE) em `SelectionMask.py`
Para contornar a não-diferenciabilidade da função `round()` (cuja derivada é zero em quase toda parte), utiliza-se o truque da identidade no backward pass:
```python
sig = torch.sigmoid(self.mask)
bin_mask = torch.round(sig).float()
# No forward pass: diff_mask = bin_mask (valores rígidos 0 ou 1)
# No backward pass: d(diff_mask)/d(mask) = d(sig)/d(mask) (gradiente flui normalmente)
diff_mask = bin_mask + (sig - sig.detach())
```

### 2. Tratamento de Unpickling em Checkpoints PyTorch
Checkpoints armazenam instâncias do modelo (`mask_model_obj`). Ao carregar checkpoints antigos com `torch.load()`, o PyTorch exige que a classe `SelectionMask` esteja no `sys.path`. Todos os scripts de utilidade e treino devem incluir:
```python
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(project_root, 'src/models'))
sys.path.append(os.path.join(project_root, 'src/utils'))
```

---

## 📌 Próximos Registros / Backlog de Tasks
* [ ] Validação de novos pipelines de compressão pré-DataLoader.
* [ ] Testes em datasets adicionais de imagens médicas / satélite.
* [ ] Extensões da biblioteca ReduMask.
