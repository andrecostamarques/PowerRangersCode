# Detalhamento Específico: Integração e Validação dos Datasets Food-101 e CIFAKE / CIFAR-10

> **Documento de Execução da Próxima Task**  
> **Localização:** `journal-docs/next-task-datasets.md`  
> **Objetivo:** Guia passo a passo para integrar os novos datasets, rodar as 3 etapas de experimentos (Baseline, Learnable, Consensus) e consolidar os resultados.

---

## 📋 PASSO A PASSO DA EXECUÇÃO

### 🔹 Passo 1: Atualização do `src/utils/DatasetsDict.py`
- [x] Adicionar suporte ao `Food101` utilizando `torchvision.datasets.Food101` com `Resize((256, 256))` e normalização RGB padrão da ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- [x] Confirmar e utilizar a instância existente do `CIFAR-10` (`datasets.CIFAR10`) no `DatasetsDict.py`.
- [x] Limpeza de arquivos temporários mantendo o diretório raiz limpo.

### 🔹 Passo 2 & 3: Automação dos Experimentos (Food-101 e CIFAR-10)
- [x] Scripts de execução sequencial criados (`run_all_food101.sh` e `run_all_cifar10.sh`).
- [x] Executar pipeline completo do Food-101 (Baseline -> Dinâmico -> Consenso -> Retreino).
- [ ] Executar pipeline completo do CIFAR-10 (Baseline -> Dinâmico -> Consenso -> Retreino).

### 🔹 Passo 4: Extração e Consolidação de Métricas
- [x] Atualizar `src/scripts/extract_metrics.py` para suportar filtros por dataset (`--dataset food101 / galaxy10`).
- [x] Gerar tabelas Markdown de resultados comparativos (Val Accuracy, F1-Score, % Pixels Ativos, $\Delta$ vs Baseline).
- [x] Registrar os resultados no `journal-docs/journal.md` e atualizar o `journal-docs/baseline-powerrangers.md`.
