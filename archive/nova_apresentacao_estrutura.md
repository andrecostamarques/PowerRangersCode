# Estrutura Proposta para a Nova Apresentação (Self-Explanatory TCC)

Esta estrutura foi pensada para criar uma apresentação "solta" (autônoma), onde cada slide funciona como um relatório curto e conciso. Ela é dividida logicamente em Contexto, Proposta (com as duas hipóteses bem explícitas), Metodologia (reaproveitando os diagramas) e Resultados/Conclusões.

---

## 📌 Slide 1: Capa (Título do Projeto)
*   **Título:** REDUMASK* OPTIMIZATION
*   **Subtítulo:** Otimização de Datasets e Modelos de Redes Neurais via Máscaras de Redução Espacial Colaborativas.
*   **Elementos:** Nome do aluno, nome do orientador, instituição e data.

---

## 📌 Slide 2: Contexto e Motivação (O Tema)
*   **Título:** O Custo do "Big Data" no Aprendizado Profundo
*   **Contexto:** O constante crescimento do volume de dados em Visão Computacional (ex: astronomia, imagens médicas) gera grandes desafios de infraestrutura.
*   **O Gargalo:**
    *   **Armazenamento e Banda Larga:** Armazenar datasets massivos e transferi-los na rede em treinamentos distribuídos gera custos e latência de rede.
    *   **Excesso de Ruído:** Em datasets de classificação de imagem, uma porção significativa da imagem de entrada é composta por fundo (pixels pretos/neutros) ou ruído irrelevante que não contribui para a decisão da rede.
*   **A Oportunidade:** Simplificar as imagens de entrada, retendo apenas o corpo morfológico essencial do objeto.

---

## 📌 Slide 3: A Proposta e as Duas Hipóteses
*   **Título:** Proposta Científica: Máscaras de Redução Espacial
*   **Ideia Central:** Introduzir um framework que aprende uma máscara binária estática de redução espacial aplicada diretamente sobre a imagem de entrada.
*   **As Duas Hipóteses de Trabalho:**
    1.  **Hipótese da Banda Larga (Eficiência de Recursos):** Ao aplicar uma máscara que zera a maioria dos pixels da imagem, reduzimos a dimensionalidade e a entropia da entrada. Isso viabiliza compressões extremas dos dados, minimizando largura de banda em redes distribuídas e transferência CPU-GPU.
    2.  **Hipótese de Desempenho (Regularização Espacial):** Ao forçar o modelo a ignorar o ruído periférico das imagens, a máscara atua como um regularizador espacial, prevenindo overfitting e estabilizando (ou até melhorando) a performance dos classificadores.

---

## 📌 Slide 4: Metodologia - Pipeline de Treinamento
*   **Título:** Fluxo Metodológico do Framework
*   **Elemento Visual:** *Reaproveitar o Diagrama 1 (Fluxograma de Treinamento: Treinamento Baseline $\rightarrow$ Geração de Máscaras Dinâmicas $\rightarrow$ Máscara Consenso $\rightarrow$ Validação de Performance).*
*   **Explicação Textual (Autônoma):**
    *   **Etapa 1 (Baseline):** Treinamento dos modelos puros para estabelecer referências de acurácia.
    *   **Etapa 2 (Learnable Mask):** Treinamento conjunto do classificador + máscara (usando STE - *Straight-Through Estimator*) para aprender dinamicamente o que é relevante em cada arquitetura.
    *   **Etapa 3 (Consenso):** Criação de uma máscara unificada através da votação majoritária entre todas as arquiteturas (filtrando os vieses individuais de cada modelo).
    *   **Etapa 4 (Classificador com Máscara Fixa):** Treino final do classificador puro com a máscara de consenso congelada na entrada.

---

## 📌 Slide 5: Metodologia - Módulos e Arquitetura do Software
*   **Título:** Arquitetura de Software e Orquestração
*   **Elemento Visual:** *Reaproveitar o Diagrama 2 (Diagrama de interação de módulos: TrainingConfig, SelectionMask, StaticMaskTraining, etc.).*
*   **Explicação Textual (Autônoma):**
    *   **SelectionMask:** Camada de máscara binária parametrizada inserida logo após a leitura do dataset.
    *   **StaticMaskTraining:** Orquestrador principal que gerencia as perdas da rede (Loss de classificação + Loss de esparsorização de máscara).
    *   **ConsensusMaskGenerator:** Módulo responsável por unificar matematicamente as máscaras individuais pós-treino em um único arquivo de consenso `.pt`.

---

## 📌 Slide 6: Resultados - Baseline vs. Máscaras Individuais (Pruning Autônomo)
*   **Título:** Resultados I: O Impacto do Descarte Individual de Pixels
*   **Tabela Comparativa:** (Baseline vs. Learnable Mask - LeNet256, ResNet20, ResNet34 e SimpleCNN).
*   **Mensagem Chave (Insight):**
    *   **Regularização da ResNet20:** Ao remover **67.3%** da imagem (ruído de fundo), a ResNet20 obteve um ganho de **+0.13% de acurácia**, comprovando a hipótese de regularização.
    *   **O Problema do Super Pruning (ResNet34):** A ResNet34 esparsorizou a máscara de forma excessiva (**0.19% de pixels ativos**), causando colapso de performance ($-14.91\%$ de acurácia). Isso mostra que modelos complexos tendem a simplificar demais as entradas se não houver restrição.

---

## 📌 Slide 7: Resultados - Baseline vs. Máscara de Consenso (A Solução Coletiva)
*   **Título:** Resultados II: Estabilidade com a Máscara de Consenso (10.11% de Pixels)
*   **Tabela Comparativa:** (Baseline vs. Consensus Mask - todos os modelos avaliados com a mesma máscara estática de 10.11% de pixels ativos).
*   **Mensagem Chave (Insight):**
    *   **Resgate da ResNet34:** A Consensus Mask recuperou o modelo do colapso, restaurando a acurácia para **85.59%** (queda de apenas $-0.06\%$) e superando a baseline original em F1-Score (**84.43%** vs. **84.06%**).
    *   **Filtro Qualitativo:** Prova de que a interseção das características consideradas relevantes por múltiplas redes distintas retém a informação mais robusta do objeto.

---

## 📌 Slide 8: Resultados - Máscara Dinâmica vs. Máscara de Consenso
*   **Título:** Resultados III: Inteligência Coletiva vs. Aprendizado Individual
*   **Tabela Comparativa:** (Comparação direta de Acurácia: Máscara Dinâmica vs. Máscara de Consenso).
*   **Mensagem Chave (Insight):**
    *   **Melhoria de Performance:** Em 3 dos 4 modelos (LeNet256, ResNet34, SimpleCNN), a máscara de consenso gerou acurácia **superior** à máscara individual.
    *   **O Caso do SimpleCNN (+3.13%):** A rede SimpleCNN performou muito melhor na máscara de consenso do que na sua própria máscara aprendida, provando o efeito benéfico da herança de features espaciais de modelos mais complexos (ResNet34).

---

## 📌 Slide 9: Resultados - Análise Visual Espacial
*   **Título:** Representação Visual da Máscara de Consenso (RGB)
*   **Elemento Visual:** *Grade 3x4 (Regimes vs. Modelos) gerada pelo script `plot_all_masks.py`.*
*   **Mensagem Chave (Insight):**
    *   Visualmente, a **Consensus Mask** desenha uma elipse nítida ao redor do corpo morfológico das galáxias.
    *   Isso comprova que o framework de fato isola a estrutura semântica essencial e descarta os pixels escuros das bordas, reduzindo a largura de banda necessária sem perda de poder preditivo.

---

## 📌 Slide 10: Discussão: Variabilidade e Robustez do Consenso
*   **Título:** Discussão: Variabilidade de Resultados e Robustez do Consenso
*   **Mensagem Chave (Insight):**
    *   **Variabilidade da Acurácia entre Testes:** 
        *   Os resultados de desempenho (acurácia) com as máscaras parecem não ser perfeitamente constantes entre diferentes execuções.
        *   *Evidência:* No teste anterior, o uso de máscaras demonstrou um desempenho superior ao baseline em mais cenários [inserir dados específicos do teste anterior aqui]. Nesta rodada, observaram-se pequenas oscilações de desempenho (ex: leves perdas de acurácia em alguns modelos).
        *   *Análise:* A convergência conjunta da máscara (via STE) e do classificador tem natureza estocástica, o que pode influenciar a estabilidade fina do resultado final de acurácia.
    *   **Robustez da Economia (Banda/Armazenamento):**
        *   Apesar da sutil oscilação de acurácia, a economia e compressão de dados são constantes e garantidas: descartar ~90% dos pixels (reduzindo a entrada a apenas 10.11%) com perdas muito pequenas de performance.
    *   **Estabilidade da Máscara de Consenso:**
        *   A melhoria da **Máscara de Consenso** em relação às **Máscaras Individuais Criadas (Dinâmicas)** é sólida e geral. Ela protege contra colapsos de esparsorização (como os 0.19% da ResNet34) e herda boas representações espaciais (trazendo +3.13% ao SimpleCNN).

---

## 📌 Slide 11: Conclusão e Validação das Hipóteses
*   **Título:** Conclusões e Validação das Hipóteses
*   *Validação das duas premissas iniciais do projeto:*
    1.  **Validação da Hipótese de Banda Larga/Armazenamento:** Provado. A máscara consensual reduziu a entrada a **10.11%** (descarte de ~90% da imagem) com perdas insignificantes de desempenho (max. $-2.63\%$ no SimpleCNN e apenas $-0.06\%$ na ResNet34). Isso viabiliza a compressão de entrada prévia no dataset.
    2.  **Validação da Hipótese de Desempenho:** Parcialmente provado/Condicional. A máscara atua como um regularizador espacial útil (melhorando +3.13% no SimpleCNN e resgatando a ResNet34), mas apresenta sensibilidade/variabilidade na acurácia dependendo do ciclo de treinamento.
*   **Trabalhos Futuros:** Desenvolvimento de biblioteca/API automatizada para pré-processamento de datasets no pipeline do DataLoader e testes em outros domínios de imagem.

