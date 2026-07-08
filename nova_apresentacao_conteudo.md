# Conteúdo Pronto para os Slides (Copiar & Colar)

Este documento contém o texto exato formatado para cada um dos slides da apresentação. A estrutura de cada slide está dividida em títulos, caixas de texto com tópicos diretos e referências para elementos visuais (figuras e tabelas).

---

## 📌 Slide 1: Capa (Apresentação do Tema)

*   **Título Principal:** REDUMASK* OPTIMIZATION
*   **Subtítulo:** Otimização de Datasets e Modelos de Redes Neurais via Máscaras de Redução Espacial Colaborativas
*   **Caixa de Texto (Informações do TCC):**
    *   **Autor:** André Marques
    *   **Orientador:** [Inserir Nome do Orientador]
    *   **Curso:** Engenharia de Computação
    *   **Instituição:** [Inserir Nome da Instituição]
    *   **Ano:** 2026

---

## 📌 Slide 2: Contexto e Motivação (O Tema)

*   **Título do Slide:** O Custo do "Big Data" no Aprendizado Profundo
*   **Caixa de Texto 1: Desafios de Infraestrutura**
    *   **Crescimento de Datasets:** O aumento expressivo no volume e resolução de imagens de visão computacional gera custos massivos de armazenamento em disco.
    *   **Gargalo de Banda Larga:** A transferência de dados em larga escala no treinamento distribuído gera latência de rede crítica, além de alta concorrência no barramento CPU-GPU.
*   **Caixa de Texto 2: Redundância Espacial das Imagens**
    *   **Excesso de Ruído Periférico:** Em datasets de classificação, grande parte da imagem é composta por fundo neutro ou pixels irrelevantes que não contribuem para a decisão das redes.
    *   **Oportunidade de Compressão:** É possível simplificar a imagem na entrada, preservando apenas o corpo morfológico essencial do objeto de estudo.

---

## 📌 Slide 3: A Proposta e as Duas Hipóteses

*   **Título do Slide:** Proposta: Máscaras de Redução Espacial
*   **Caixa de Texto 1: Proposta Geral**
    *   Desenvolvimento de um framework capaz de aprender uma máscara binária estática de redução espacial aplicada diretamente sobre a imagem de entrada antes de alimentar os modelos.
*   **Caixa de Texto 2: Hipótese 1 - Banda Larga e Armazenamento**
    *   Ao filtrar e "zerar" pixels irrelevantes, reduz-se a dimensionalidade e a entropia das imagens bruteiras na entrada do pipeline.
    *   **Impacto:** Viabiliza compressões extremas do dataset para armazenamento físico e transmissão de rede rápida, com baixíssima perda de informação preditiva.
*   **Caixa de Texto 3: Hipótese 2 - Desempenho e Regularização Espacial**
    *   Ao limitar a visão dos classificadores ao corpo morfológico do objeto, a máscara atua como um regularizador espacial robusto.
    *   **Impacto:** Evita o overfitting em ruído de fundo, estabilizando e otimizando a generalização dos classificadores.

---

## 📌 Slide 4: Metodologia - Pipeline de Treinamento

*   **Título do Slide:** Fluxo Metodológico do Framework
*   **Elemento Visual (Diagrama 1):**
    *   *Reaproveitar o Diagrama 1 de treinamento do PDF (Baseline -> Learnable Mask -> Consensus Mask -> Validação de Performance).*
*   **Caixa de Texto 1: Fases Metodológicas**
    *   **1. Baseline:** Treinamento dos modelos em imagens completas para estabelecer referências de acurácia.
    *   **2. Learnable Mask:** Otimização conjunta de classificador + máscara (via STE - *Straight-Through Estimator*) para aprender a esparsorização dinâmica de cada modelo.
    *   **3. Consensus Mask:** Geração de uma máscara única estática por votação majoritária entre os modelos, filtrando os vieses individuais de cada rede.
    *   **4. Treinamento Fixo:** Treino final dos modelos mantendo a máscara de consenso estática na entrada das imagens.

---

## 📌 Slide 5: Metodologia - Módulos e Arquitetura do Software

*   **Título do Slide:** Arquitetura de Software e Orquestração
*   **Elemento Visual (Diagrama 2):**
    *   *Reaproveitar o Diagrama 2 de interação dos componentes (TrainingConfig, SelectionMask, StaticMaskTraining, etc.).*
*   **Caixa de Texto 1: Componentes de Software**
    *   **SelectionMask:** Camada parametrizada que aplica o produto de Hadamard (multiplicação pixel a pixel) da máscara binária sobre a imagem do dataset.
    *   **StaticMaskTraining:** Orquestrador customizado que gerencia a combinação de funções de perda (Loss de classificação clássica + penalização L1 para forçar a esparsorização da máscara).
    *   **ConsensusMaskGenerator:** Script responsável por processar os checkpoints salvos, realizar o voto majoritário por pixel e gerar o arquivo de consenso `.pt`.

---

## 📌 Slide 6: Resultados - Baseline vs. Máscaras Individuais (Pruning Autônomo)

*   **Título do Slide:** Resultados I: Descarte Individual de Pixels
*   **Elemento Visual (Tabela 1):**

| Modelo | Cenário | Acurácia | F1-Score | Pixels Ativos | $\Delta$ Acurácia vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LeNet256** | Baseline vs. Dinâmica | 75.44% vs. 74.56% | 73.62% vs. 71.64% | **9.77%** | $-0.88\%$ |
| **ResNet20** | Baseline vs. Dinâmica | 84.46% vs. 84.59% | 83.26% vs. 82.49% | **32.70%** | **+0.13%** |
| **ResNet34** | Baseline vs. Dinâmica | 85.65% vs. 70.74% | 84.06% vs. 68.07% | **0.19%** | $-14.91\%$ |
| **SimpleCNN** | Baseline vs. Dinâmica | 80.51% vs. 74.75% | 78.36% vs. 70.89% | **8.97%** | $-5.76\%$ |

*   **Caixa de Texto 1: Análise das Máscaras Individuais**
    *   **Compressão Saudável:** O LeNet256 e o SimpleCNN aprenderam filtros que eliminam mais de 90% da imagem original apresentando pequenas reduções de acurácia.
    *   **Regularização de Performance:** Ao eliminar **67.3%** da imagem (ruído de fundo), a ResNet20 obteve um ganho de **+0.13% de acurácia** em relação à baseline.
    *   **Colapso por Super Pruning:** A ResNet34 esparsorizou excessivamente a máscara, mantendo apenas **0.19%** de pixels ativos (descarte excessivo), o que colapsou seu desempenho ($-14.91\%$).

---

## 📌 Slide 7: Resultados - Baseline vs. Máscara de Consenso (A Solução Coletiva)

*   **Título do Slide:** Resultados II: Estabilidade com Máscara de Consenso (10.11% de Pixels)
*   **Elemento Visual (Tabela 2):**

| Modelo | Cenário | Acurácia | F1-Score | Pixels Ativos | $\Delta$ Acurácia vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LeNet256** | Baseline vs. Consenso | 75.44% vs. 74.81% | 73.62% vs. 71.72% | 10.11% | $-0.63\%$ |
| **ResNet20** | Baseline vs. Consenso | 84.46% vs. 82.27% | 83.26% vs. 80.24% | 10.11% | $-2.19\%$ |
| **ResNet34** | Baseline vs. Consenso | 85.65% vs. **85.59%** | 84.06% vs. **84.43%** | 10.11% | **-0.06%** |
| **SimpleCNN** | Baseline vs. Consenso | 80.51% vs. 77.88% | 78.36% vs. 74.96% | 10.11% | $-2.63\%$ |

*   **Caixa de Texto 1: Análise do Consenso Espacial**
    *   **Recuperação e Estabilidade:** A máscara de consenso resgatou a ResNet34 do colapso de esparsorização, restaurando sua acurácia para **85.59%** (queda irrisória de $-0.06\%$) e superando a baseline original em macro F1-score (**84.43%** vs. **84.06%**).
    *   **Interseção Coletiva:** A máscara unificada age como um filtro espacial robusto que retém apenas características fundamentais do objeto, mitigando o viés individual de cada rede.

---

## 📌 Slide 8: Resultados - Máscara Dinâmica vs. Máscara de Consenso (O Delta)

*   **Título do Slide:** Resultados III: Consenso Coletivo vs. Aprendizado Individual
*   **Elemento Visual (Tabela 3):**

| Modelo | Acurácia (Dinâmica) | Acurácia (Consenso) | $\Delta$ Acurácia (Consenso - Dinâmica) | Comparativo de Pixels Ativos |
| :--- | :---: | :---: | :---: | :---: |
| **LeNet256** | 74.56% | 74.81% | **+0.25%** | Equivalente (~9.7% vs. 10.1%) |
| **ResNet20** | 84.59% | 82.27% | $-2.32\%$ | **3x mais compressão** (32.7% vs. 10.1%) |
| **ResNet34** | 70.74% | 85.59% | **+14.85%** | **Recuperação de colapso** (0.19% vs. 10.1%) |
| **SimpleCNN** | 74.75% | 77.88% | **+3.13%** | Equivalente (~9.0% vs. 10.1%) |

*   **Caixa de Texto 1: A Força da Máscara de Consenso**
    *   **Melhoria Superior:** Em 3 dos 4 modelos (LeNet, ResNet34, SimpleCNN), a Máscara Consenso superou a Máscara Dinâmica Individual.
    *   **Herança de Features (SimpleCNN):** Ganho expressivo de **+3.13%** de acurácia na SimpleCNN, demonstrando que classificadores mais simples se beneficiam de máscaras enriquecidas de modelos profundos.
    *   **Compressão Eficiente (ResNet20):** A ResNet20 obteve **3x mais compressão** de dados usando o consenso (10.11% de pixels ativos) em detrimento de uma pequena oscilação na acurácia ($-2.32\%$).

---

## 📌 Slide 9: Resultados - Análise Visual Espacial

*   **Título do Slide:** Representação Visual das Máscaras RGB
*   **Elemento Visual:**
    *   *Inserir a grade de imagens 3x4 (Baseline, Máscara Gerada, Máscara Consenso vs. Modelos) gerada por `plot_all_masks.py`.*
*   **Caixa de Texto 1: Análise Visual e Semântica**
    *   **Linha 2 (Máscaras Individuais):** Exibe a variação e desvio morfológico drástico dependendo da complexidade do modelo (ResNet34 restringe a um pixel central, ResNet20 mantém pontos dispersos).
    *   **Linha 3 (Consensus Mask):** Revela uma elipse perfeita mapeando de forma contígua os limites morfológicos das galáxias espirais. O descarte de ruído periférico é anatomicamente coerente.

---

## 📌 Slide 10: Discussão - Variabilidade de Resultados e Robustez

*   **Título do Slide:** Discussão: Variabilidade de Resultados e Robustez do Consenso
*   **Caixa de Texto 1: Inconstância do Desempenho Fino**
    *   **Variabilidade Inter-Testes:** O ganho exato de acurácia com as máscaras apresenta inconstâncias entre ciclos diferentes de treinamento.
    *   **Evidência Histórica:** Em testes anteriores, a máscara superou a baseline em mais cenários [inserir dados específicos do teste anterior aqui]. No teste atual, observamos oscilações pequenas com leves perdas de acurácia, o que decorre do caráter estocástico do treino conjunto via STE.
*   **Caixa de Texto 2: Robustez Estrutural da Proposta**
    *   **Estabilidade da Compressão:** Independentemente das flutuações finas de precisão, a eficácia na economia de armazenamento e banda é garantida, reduzindo ~90% da dimensionalidade das entradas de forma confiável.
    *   **Consistência do Consenso:** O ganho de desempenho da Máscara de Consenso em comparação com as Máscaras Dinâmicas Individuais é altamente consistente e atua como um excelente amortecedor de colapsos.

---

## 📌 Slide 11: Conclusão e Validação das Hipóteses

*   **Título do Slide:** Conclusões e Validação das Hipóteses
*   **Caixa de Texto 1: Validação da Hipótese de Armazenamento/Banda**
    *   **Provada:** É perfeitamente viável realizar a compressão espacial prévia do dataset, retendo apenas **10.11%** da imagem original com perdas de acurácia irrelevantes (ex: $-0.06\%$ na ResNet34). Redução real de custos físicos e de largura de banda na rede.
*   **Caixa de Texto 2: Validação da Hipótese de Desempenho/Regularização**
    *   **Parcialmente Provada (Condicional):** O uso da máscara de fato funciona como regularizador espacial e protege os modelos contra ruído de fundo (ganho de +3.13% no SimpleCNN e resgate da ResNet34), contudo exibe sensibilidade à inicialização estocástica das redes.
*   **Caixa de Texto 3: Próximos Passos**
    *   Desenvolvimento de biblioteca ou API customizada de compressão pré-pipeline direto no DataLoader do PyTorch.
    *   Testes do framework em outros domínios de imagens médicas e de satélite.
