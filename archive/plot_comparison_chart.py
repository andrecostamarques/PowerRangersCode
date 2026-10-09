#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

def generate_comparison_plot():
    # -------------------------------------------------------------
    # Configuração dos dados
    # -------------------------------------------------------------
    models = ['LeNet256', 'ResNet20', 'ResNet34', 'SimpleCNN']
    regimes = ['Baseline', 'Learnable Mask', 'Consensus Mask']
    
    # Cores modernas (Aesthetics premium)
    # Azul escuro/cobalto para Baseline, Coral/Laranja para Learnable, Verde esmeralda para Consensus
    colors = {
        'Baseline': '#2b5c8f',       # Muted Cobalt Blue
        'Learnable Mask': '#d97724',  # Sleek Amber/Orange
        'Consensus Mask': '#2a9d8f'   # Teal/Emerald Green
    }

    # Dados de Acurácia (%)
    accuracy = {
        'Baseline': [75.44, 84.46, 85.65, 80.51],
        'Learnable Mask': [74.56, 84.59, 70.74, 74.75],
        'Consensus Mask': [74.81, 82.27, 85.59, 77.88]
    }

    # Dados de F1-Score (%)
    f1_score = {
        'Baseline': [73.62, 83.26, 84.06, 78.36],
        'Learnable Mask': [71.64, 82.49, 68.07, 70.89],
        'Consensus Mask': [71.72, 80.24, 84.43, 74.96]
    }

    # Configuração da grade de plot (1 linha, 2 colunas para Acurácia e F1-Score)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.patch.set_facecolor('#fafafa')  # Fundo sutilmente off-white

    x = np.arange(len(models))  # Posições dos grupos (classificadores)
    width = 0.25                # Largura das barras

    # -------------------------------------------------------------
    # Subplot 1: Acurácia de Validação
    # -------------------------------------------------------------
    ax_acc = axes[0]
    ax_acc.set_facecolor('#ffffff')
    ax_acc.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    # Plotando as barras
    rects1 = ax_acc.bar(x - width, accuracy['Baseline'], width, label='Baseline (100% Pixels)', color=colors['Baseline'], zorder=3)
    rects2 = ax_acc.bar(x, accuracy['Learnable Mask'], width, label='Learnable Mask (Dinâmica)', color=colors['Learnable Mask'], zorder=3)
    rects3 = ax_acc.bar(x + width, accuracy['Consensus Mask'], width, label='Consensus Mask (10.11% Pixels)', color=colors['Consensus Mask'], zorder=3)

    ax_acc.set_title('Acurácia de Validação (%) por Modelo', fontsize=14, fontweight='bold', pad=15, color='#333333')
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(models, fontsize=12, fontweight='bold', color='#444444')
    ax_acc.set_ylabel('Porcentagem (%)', fontsize=12, color='#333333')
    ax_acc.set_ylim(0, 100)

    # Adicionar os valores no topo das barras
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#444444')

    autolabel(rects1, ax_acc)
    autolabel(rects2, ax_acc)
    autolabel(rects3, ax_acc)

    # -------------------------------------------------------------
    # Subplot 2: F1-Score
    # -------------------------------------------------------------
    ax_f1 = axes[1]
    ax_f1.set_facecolor('#ffffff')
    ax_f1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    rects4 = ax_f1.bar(x - width, f1_score['Baseline'], width, label='Baseline (100% Pixels)', color=colors['Baseline'], zorder=3)
    rects5 = ax_f1.bar(x, f1_score['Learnable Mask'], width, label='Learnable Mask (Dinâmica)', color=colors['Learnable Mask'], zorder=3)
    rects6 = ax_f1.bar(x + width, f1_score['Consensus Mask'], width, label='Consensus Mask (10.11% Pixels)', color=colors['Consensus Mask'], zorder=3)

    ax_f1.set_title('F1-Score (%) por Modelo', fontsize=14, fontweight='bold', pad=15, color='#333333')
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(models, fontsize=12, fontweight='bold', color='#444444')
    ax_f1.set_ylim(0, 100)

    autolabel(rects4, ax_f1)
    autolabel(rects5, ax_f1)
    autolabel(rects6, ax_f1)

    # Legenda única centralizada no topo
    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=True, facecolor='#ffffff', fontsize=11)

    plt.suptitle('Comparativo Geral de Métricas: Acurácia vs. F1-Score', fontsize=16, fontweight='bold', y=1.02, color='#222222')
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    # Salvar a imagem no workspace
    output_path = 'plot_comparativo_metricas.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico comparativo gerado e salvo com sucesso em: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    generate_comparison_plot()
