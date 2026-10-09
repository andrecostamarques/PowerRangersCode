# Baseline & Visão Geral do Projeto PowerRangersCode

> **Documento de Referência Contextual para Desenvolvedores e Agentes de IA**
> **Localização:** `journal-docs/baseline-powerrangers.md`
> **Projeto:** Otimização de Datasets e Modelos via Máscaras de Redução Espacial Colaborativas (ReduMask)

---

## 📌 1. Visão Geral e Hipóteses do Projeto

O **PowerRangersCode** é um framework em PyTorch para a **redução de dimensionalidade espacial e regularização de modelos de Aprendizado Profundo** em tarefas de Visão Computacional (ex: dataset astronômico **Galaxy10**). 

A abordagem utiliza um módulo customizado de **Máscara de Seleção Binária Aprendível** (*SelectionMask*) otimizada via **Straight-Through Estimator (STE)** em conjunto com o classificador de imagem.

### As Duas Hipóteses Científicas:
1. **Hipótese 1 — Eficiência de Banda Larga e Armazenamento (Compressão Espacial):**
   * *Premissa:* Grande parte dos pixels de imagens de entrada (especialmente em astrofotografia ou exames médicos) é composta por fundo escuro/neutro redundante.
   * *Objetivo:* Aprender um filtro binário estático que descarte ~90% dos pixels periféricos antes da transmissão ou processamento, mantendo o poder preditivo do modelo e economizando largura de banda e armazenamento.
2. **Hipótese 2 — Desempenho e Regularização Espacial:**
   * *Premissa:* Ao "cegar" o modelo para o ruído e fundos irrelevantes, a máscara atua como um regularizador espacial robusto.
   * *Objetivo:* Mitigar overfitting em modelos complexos e melhorar a generalização em modelos mais simples (permitindo a herança de features espaciais).

---

## 🛠️ 2. Estrutura do Repositório e Módulos do Sistema

```
PowerRangersCode/
├── journal-docs/             # Documentação histórica e contextos para agentes/equipe
│   └── baseline-powerrangers.md
├── src/                      # Código-fonte principal
│   ├── models/               # Arquiteturas de classificadores e módulo SelectionMask
│   │   ├── SelectionMask.py  # Máscara binária treinável via STE
│   │   ├── LeNet256.py       # LeNet adaptada para resolução 256x256
│   │   ├── ResNet20.py       # ResNet-20 customizada
│   │   ├── SimpleCNNRGB.py   # CNN leve para 3 canais RGB
│   │   └── LeNet5.py / LeNet5RGB.py / SimpleCNN.py
│   ├── utils/                # Utilitários de treino, loss e dataset
│   │   ├── TrainingConfig.py # Gerenciador central de hiperparâmetros
│   │   ├── DatasetsDict.py   # Carregador e transformador de datasets (Galaxy10, MNIST, etc.)
│   │   ├── CustomDatasets.py # Wrappers customizados de datasets PyTorch
│   │   ├── StaticMaskTraining.py # Orquestrador do treino conjunto (Modelo + Máscara)
│   │   ├── TotalLoss.py      # Combinação de Loss de Classificação + λ * Loss de Esparsidade
│   │   ├── LambdaScheduler.py# Ajuste dinâmico do fator de esparsidade λ
│   │   └── ModelTester.py    # Avaliador de checkpoints e matriz de confusão
│   └── scripts/              # Scripts de execução e análise
│       ├── training_loop.py  # Exemplo de ponto de entrada para treino dinâmico
│       ├── train_classifier.py # Treinador flexível (Baseline ou Máscara Estática Congelada)
│       ├── generate_consensus_mask.py # Gerador da Máscara de Consenso (Votação Majoritária)
│       ├── extract_metrics.py# Extração automatizada de acurácia, F1-score e esparsidade
│       ├── plot_all_masks.py # Renderização visual 3x4 em RGB das máscaras aprendidas
│       └── run_experiments.py# Automação de suíte completa de experimentos
├── checkpoints/              # Checkpoints (.pt) de modelos e máscaras geradas
├── notebooks/                # Notebooks Jupyter para análise e plots interativos
├── environment.yml           # Especificação do ambiente Conda (PowerRanger-env)
└── README.md                 # Guia de instalação e visão rápida
```

---

## ⚙️ 3. Funcionamento Interno dos Módulos Principais

### A. `SelectionMask.py` (Módulo STE)
* **Objetivo:** Aprender uma máscara binária estática $M \in \{0, 1\}^{C \times H \times W}$ multiplicada via produto de Hadamard ($X_{\text{masked}} = X \odot M$).
* **Straight-Through Estimator (STE):**
  ```python
  sig = torch.sigmoid(self.mask)
  bin_mask = torch.round(sig).float()
  diff_mask = bin_mask + (sig - sig.detach()) # STE Trick: forward usa bin_mask, backward usa gradiente do sigmoid
  ```
* **Loss de Esparsidade ($L1$):** `mask_l1_loss` calcula a proporção de pixels ativos (1s) sobre o total de pixels (`sum() / numel()`).

### B. `TotalLoss.py` & `LambdaScheduler.py`
* **Cálculo da Perda Total:**
  $$\text{Total Loss} = \text{Classification Loss} + \lambda \cdot \text{Mask Sparsity Loss}$$
* **Scheduler Adaptativo ($\lambda$):** O `LambdaScheduler` monitora a redução da perda. Se a perda estagnar por `patience` épocas acima do `treshold`, o fator $\lambda$ é multiplicado por `factor` para forçar maior compressão espacial.

### C. `StaticMaskTraining.py` (Treino Conjunto)
* Orquestra o treinamento dinâmico onde **ambos** o classificador e a `SelectionMask` atualizam seus parâmetros simultaneamente.
* Salva checkpoints `.pt` contendo os dicionários de estado do modelo, da máscara, otimizador, acurácia, matriz de confusão e valor de $\lambda$.

### D. `train_classifier.py` (Treino com Máscara Fixa / Baseline)
* Usado para treinar o classificador **puro** sem máscara (Baseline) ou com uma máscara **estática congelada** na entrada (ex: `consensus_mask.pt`).
* Quando `mask_checkpoint_path` é fornecido, a máscara é carregada, colocada em `eval()` e seus parâmetros são totalmente congelados (`requires_grad = False`).

### E. `generate_consensus_mask.py` (Inteligência Coletiva)
* **Objetivo:** Mitigar a variação estocástica e o viés espacial de modelos individuais (ex: colapso da ResNet34 para 0.19% de pixels) gerando uma máscara consenso unificada.
* **Passos do Algoritmo:**
  1. Extrai as máscaras binárias dos 4 modelos ($M_i = \text{round}(\sigma(W_i))$).
  2. Média element-wise: $\bar{M} = \frac{1}{4} \sum_{i=1}^4 M_i$.
  3. Votação majoritária ($\text{limiar} \ge 0.5$): $M_{\text{consenso}} = \mathbb{I}(\bar{M} \ge 0.5)$.
  4. Codificação de Logits para PyTorch:
     $$W_{\text{consenso}} = \begin{cases} +10.0 & \text{se } M_{\text{consenso}} = 1.0 \quad (\sigma(10.0) \approx 1.0) \\ -10.0 & \text{se } M_{\text{consenso}} = 0.0 \quad (\sigma(-10.0) \approx 0.0) \end{cases}$$
* Gera o arquivo unificado `checkpoints/consensus_mask.pt` com **10.11% de pixels ativos** (descarte de 89.89% da imagem de entrada).

---

## 🔄 4. Três Regimes de Treinamento

| Regime | Descrição | Otimização | Esparsidade Resultante |
| :--- | :--- | :--- | :---: |
| **1. Baseline** | Treinamento padrão do classificador puro na imagem bruta completa. | Apenas pesos do classificador. | 100.00% (Sem máscara) |
| **2. Learnable Mask (Dinâmica)** | Treinamento conjunto (Classificador + `SelectionMask` via STE e $\lambda$-scheduler). | Pesos do classificador + parâmetros da máscara. | Varia por modelo (0.19% a 32.70%) |
| **3. Consensus Mask** | Treinamento do classificador puro utilizando a máscara de consenso fixada na entrada. | Apenas pesos do classificador (Máscara congelada). | **10.11%** (Fixo para todos) |

---

### 5.1 Dataset Galaxy10

Resultados extraídos via `src/scripts/extract_metrics.py --dataset galaxy10`:

| Arquitetura | Regime de Treino | Época | Val Accuracy | Precision | Recall | F1-Score | Pixels Ativos (%) | $\Delta$ Acurácia vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LeNet256** | Baseline | 168 | 75.44% | 75.41% | 72.79% | 73.62% | 100.00% | - |
| | Learnable Mask | 164 | 74.56% | 74.05% | 71.52% | 71.64% | 9.77% | -0.88% |
| | **Consensus Mask** | 192 | **74.81%** | 74.31% | 70.89% | **71.72%** | **10.11%** | **-0.63%** |
| **ResNet20** | Baseline | 184 | 84.46% | 84.28% | 82.60% | 83.26% | 100.00% | - |
| | Learnable Mask | 159 | 84.59% | 83.79% | 81.66% | 82.49% | 32.70% | **+0.13%** |
| | **Consensus Mask** | 168 | **82.27%** | 80.74% | 80.24% | **80.24%** | **10.11%** | **-2.19%** |
| **ResNet34** | Baseline | 72 | 85.65% | 84.02% | 84.38% | 84.06% | 100.00% | - |
| | Learnable Mask | 223 | 70.74% *(colapso)* | 69.88% | 68.16% | 68.07% | 0.19% | -14.91% |
| | **Consensus Mask** | 162 | **85.59%** | 85.19% | 83.81% | **84.43%** | **10.11%** | **-0.06%** |
| **SimpleCNNRGB**| Baseline | 190 | 80.51% | 80.45% | 77.42% | 78.36% | 100.00% | - |
| | Learnable Mask | 200 | 74.75% | 73.78% | 70.34% | 70.89% | 8.97% | -5.76% |
| | **Consensus Mask** | 198 | **77.88%** | 78.40% | 75.04% | **74.96%** | **10.11%** | **-2.63%** |

### 5.2 Dataset Food-101 (101 Classes)

Resultados extraídos via `src/scripts/extract_metrics.py --dataset food101`:

| Arquitetura | Regime de Treino | Época | Val Accuracy | Precision | Recall | F1-Score | Pixels Ativos (%) | $\Delta$ Acurácia vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LeNet256** | Baseline | 190 | 33.69% | 35.03% | 33.49% | 33.14% | 100.00% | - |
| | Learnable Mask | 134 | 31.59% | 32.71% | 31.35% | 30.54% | 51.53% | -2.10% |
| | **Consensus Mask** | 200 | **31.08%** | 33.02% | 31.02% | **30.28%** | **58.04%** | **-2.61%** |
| **ResNet20** | Baseline | 187 | 56.34% | 59.03% | 56.39% | 56.40% | 100.00% | - |
| | Learnable Mask | 168 | 48.70% | 52.75% | 48.57% | 48.70% | 26.15% | -7.64% |
| | **Consensus Mask** | 200 | **52.98%** | 55.71% | 52.85% | **52.32%** | **58.04%** | **-3.36%** |
| **ResNet34** | Baseline | 48 | 59.14% | 59.05% | 59.02% | 58.82% | 100.00% | - |
| | Learnable Mask | 61 | 52.65% | 55.08% | 52.60% | 52.39% | 85.81% | -6.49% |
| | **Consensus Mask** | 200 | **50.51%** | 50.23% | 50.37% | **49.95%** | **58.04%** | **-8.63%** |
| **SimpleCNNRGB**| Baseline | 189 | 17.83% | 16.41% | 17.73% | 16.53% | 100.00% | - |
| | Learnable Mask | 23 | 0.74% *(colapso)* | 0.01% | 0.99% | 0.01% | 0.00% | -17.09% |
| | **Consensus Mask** | 200 | **14.72%** | 14.96% | 14.66% | **14.27%** | **58.04%** | **-3.11%** |

### Principais Achados Científicos:
1. **Resgate do Colapso do SimpleCNNRGB no Food-101:** Na máscara dinâmica, o `SimpleCNNRGB` zerou a retenção de pixels (0.00%), resultando no colapso do modelo (0.74% de acurácia em 101 classes). A **Consensus Mask** resgatou o modelo, elevando a acurácia para **14.72%** (+13.98% vs máscara aprendida).
2. **Transferência Coletiva de Sinal na ResNet20:** A `ResNet20` aumentou sua acurácia de **48.70%** (máscara dinâmica com 26.15% de pixels) para **52.98%** na Consensus Mask (+4.28%), beneficiando-se da retenção de áreas focais acordadas pelos 4 modelos.
3. **Estabilização da Máscara de Consenso em 58.04%:** No Food-101 (101 classes de alimentos complexos), a votação majoritária conservou 58.04% dos pixels focais, descartando **41.96%** do fundo redundante.
4. **Resgate do Colapso da ResNet34 no Galaxy10:** Na máscara dinâmica do Galaxy10, a ResNet34 super-esparsorizou a máscara para 0.19%, colapsando a acurácia para 70.74%. A **Consensus Mask** resgatou o modelo para **85.59%** (diferença de apenas $-0.06\%$ vs baseline) e F1-Score superior à própria baseline (**84.43%** vs. **84.06%**).

---

## 🚀 6. Como Executar os Scripts do Projeto

### Instalação do Ambiente
```bash
conda env create -n PowerRanger-env python=3.12
conda activate PowerRanger-env
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
conda env update --file environment.yml --prune
```

### Execução dos Treinos e Análises (Sempre rodar dentro de `src/scripts/`)
```bash
# Entrar no diretório de scripts (Regra obrigatória do projeto)
cd src/scripts

# 1. Extrair tabela completa de métricas (Markdown):
python3 extract_metrics.py

# 2. Gerar a máscara de consenso a partir dos melhores checkpoints:
python3 generate_consensus_mask.py --dataset food101

# 3. Gerar a grade comparativa 3x4 visual das máscaras (RGB):
python3 plot_all_masks.py

# 4. Plotar o histórico de treino (training_log.csv):
python3 plot_training_log.py --csv_path ../../checkpoints/food101/food101_simplecnnrgb_200epochs/training_log.csv

# 5. Treinar um classificador específico com Máscara de Consenso congelada:
python3 train_classifier.py --dataset food101 --model resnet34 --mask_checkpoint ../../checkpoints/food101/consensus_mask_food101.pt --epochs 200
```

---

## 📝 7. Resumo para Agentes de IA
* Ao continuar o desenvolvimento do projeto em novos chats, este documento em `journal-docs/baseline-powerrangers.md` contém todo o contexto teórico, mapeamento de arquivos, pipelines e métricas baseline validadas.
* **Estado Atual:** O pipeline de 3 regimes (Baseline, Learnable, Consensus) está completo e validado com scripts funcionais.
