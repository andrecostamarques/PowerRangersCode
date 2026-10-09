# Lista Temporária de Tarefas (temp-todo.md)

> **Documento de Controle de Tarefas Imediatas e Backlog de Experimentos**  
> **Localização:** `journal-docs/temp-todo.md`  
> **Priorização Atualizada:** 
> 1. Validação em Novos Datasets (Food-101 e CIFAKE / CIFAR-10)
> 2. Estudo de Ablação da ResNet34 (no Consensus Mask do Galaxy10 e dos novos datasets)
> 3. Explicabilidade & Atenção Espacial (Grad-CAM)
> 4. Escrita do Artigo (Neurocomputing / JCNN)

---

## 🚀 ROADMAP DE EXECUÇÃO REORGANIZADO

---

### 📍 PRIORIDADE 1: Validação nos Datasets Food-101 e CIFAKE / CIFAR-10 (FOCO ATUAL)
- [x] **Task 1.1 — Integração de Datasets em `DatasetsDict.py`:** 
  * Adicionar `Food101` via `torchvision.datasets.Food101` com redimensionamento e transformações padronizadas.
  * Verificar e ajustar suporte ao `CIFAR-10` / `CIFAKE`.
- [x] **Task 1.2 — Experimentos no Food-101:**
  * Treino Baseline (4 modelos, sem máscara).
  * Treino de Máscara Dinâmica (4 modelos + SelectionMask).
  * Geração do `consensus_mask_food101.pt`.
  * Retreino com Consensus Mask.
- [ ] **Task 1.3 — Experimentos no CIFAKE / CIFAR-10:**
  * Treino Baseline (4 modelos, sem máscara).
  * Treino de Máscara Dinâmica (4 modelos + SelectionMask).
  * Geração do `consensus_mask_cifake.pt`.
  * Retreino com Consensus Mask.
- [x] **Task 1.4 — Tabela de Resultados Comparativa:** Extrair métricas oficiais em Markdown de todos os regimes nos novos datasets (Food-101 extraído).

---

### 📍 PRIORIDADE 2: Estudo de Ablação do ResNet34
- [ ] **Task 2.1 — Atualizar Gerador de Máscara:** Adicionar parâmetro `--exclude resnet34` em `generate_consensus_mask.py`.
- [ ] **Task 2.2 — Ablação no Galaxy10:** Gerar consenso sem ResNet34 e retreinar os 4 modelos.
- [ ] **Task 2.3 — Ablação nos Novos Datasets:** Testar consenso com e sem ResNet34 no Food-101 e CIFAKE.

---

### 📍 PRIORIDADE 3: Explicabilidade & Atenção com Grad-CAM
- [ ] **Task 3.1 — Script de Grad-CAM (`src/scripts/evaluate_gradcam.py`):** Gerar mapas de calor do Grad-CAM.
- [ ] **Task 3.2 — Correlação Espacial:** Calcular correlação IoU / Pearson entre Grad-CAM baseline e a máscara de consenso binária.

---

### 📍 PRIORIDADE 4: Escrita do Artigo (Neurocomputing / JCNN)
- [ ] **Task 4.1 — Consolidação de Figuras e Gráficos.**
- [ ] **Task 4.2 — Redação dos Resultados e Metodologia em LaTeX.**
