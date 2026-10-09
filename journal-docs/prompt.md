# Diretriz de Inicialização & Protocolo journal-docs

> **Instruções para Agentes de IA em Novas Sessões de Chat**  
> **Localização:** `journal-docs/prompt.md`  
> **Objetivo:** Estabelecer o protocolo automático de leitura de contexto e atualização contínua da documentação do projeto.

---

## 🚀 Prompt Pronto para Copiar e Colar em Novos Chats

Ao abrir um novo chat, cole o seguinte comando para o agente:

```text
Olá! Por favor, leia os arquivos no diretório `journal-docs/`:
1. `journal-docs/baseline-powerrangers.md` (Contexto técnico, código, modelos e métricas baseline)
2. `journal-docs/journal.md` (Diário de bordo com histórico de tarefas e backlog)
3. `journal-docs/prompt.md` (Protocolo de documentação contínua)

Após ler esses arquivos, faça um resumo ultra-conciso do estado atual do projeto e me confirme que está pronto para continuar seguindo este protocolo.
```

---

## 🤖 Protocolo de Atuação do Agente (journal-docs Protocol)

Se você é um Agente de IA lendo este documento, siga **rigorosamente** as seguintes regras durante todo o atendimento:

### 1. Absorção de Contexto Imediata
* Não peça ao usuário para re-explicar a arquitetura ou o histórico do projeto.
* Todo o contexto técnico (STE, SelectionMask, Consensus Mask, datasets, modelos, scripts) está em `journal-docs/baseline-powerrangers.md`.
* O histórico de tarefas executadas e o backlog atual estão em `journal-docs/journal.md`.

### 2. Protocolo de Atualização Contínua (Journaling)
Ao concluir qualquer tarefa significativa, refatoração, experimento ou criação de script:
* **Registrar a Task no `journal-docs/journal.md`:**
  * Adicionar uma nova entrada (`### 🗓️ Entry #00X — Nome da Task`).
  * Incluir: Data/Status, Objetivos, Ações Executadas & Scripts Criados, Resultados/Métricas obtidas, Artefatos Gerados e Estado do Backlog.
* **Manter a Baseline Atualizada:**
  * Se a tarefa alterar a estrutura de arquivos, a lógica principal dos módulos (`src/`) ou adicionar novos modelos/métricas oficiais, atualizar também o documento `journal-docs/baseline-powerrangers.md`.

### 3. Preservação de Qualidade e Honestidade Acadêmica
* Certifique-se de validar numericamente qualquer métrica (acurácia, F1-score, esparsidade) antes de registrar.
* Respeite os padrões visuais e scripts de plotagem existentes no projeto (`plot_comparison_chart.py`, `plot_all_masks.py`, `extract_metrics.py`).

### 4. Diretriz Estrita de Execução de Scripts
* **Localização de Execução:** TODOS os scripts da pasta `src/scripts/` devem ser rodados a partir do próprio diretório `src/scripts/` (ex: `cd src/scripts && python3 <script.py>`). Não execute `python3 src/scripts/<script.py>` a partir da raiz.

