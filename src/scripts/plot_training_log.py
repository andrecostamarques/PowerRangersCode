#!/usr/bin/env python3
"""
Plot Training Log Script
========================
Visualiza graficamente o histórico de treinamento a partir do arquivo `training_log.csv`.
Permite analisar 'no olhômetro' a melhor relação custo-benefício entre esparsidade da máscara (pixels ativos / mask loss)
e acurácia de validação ao longo das épocas.

Uso via Linha de Comando:
    python src/scripts/plot_training_log.py --csv_path checkpoints/meu_treino/training_log.csv --epoch 50
    python src/scripts/plot_training_log.py --csv_path checkpoints/meu_treino/training_log.csv --save_plot plot_log.png

Colunas esperadas no CSV:
    epoch, total_loss, model_loss, mask_loss, val_accuracy, lambda_value, lambda_patience_count
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training_log(csv_path: str, epoch: int = None, save_path: str = None, show_plot: bool = True):
    """
    Plota as perdas, acurácia e o parâmetro lambda ao longo das épocas com base no training_log.csv.

    Args:
        csv_path (str): Caminho para o arquivo `training_log.csv`.
        epoch (int, optional): Época selecionada para destacar com linha vertical. Se None, seleciona a de maior acurácia.
        save_path (str, optional): Caminho para salvar o gráfico gerado (ex: 'resultado.png').
        show_plot (bool): Se True, exibe o gráfico interativo com plt.show().
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Arquivo CSV não encontrado: {csv_path}")

    # Carrega os dados do CSV
    df = pd.read_csv(csv_path)

    # Validar colunas necessárias
    required_cols = {'epoch', 'total_loss', 'model_loss', 'mask_loss', 'val_accuracy', 'lambda_value'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"❌ Colunas ausentes no CSV: {missing_cols}")

    # Determinar a época global selecionada
    if epoch is None:
        # Por padrão, seleciona a época com melhor acurácia de validação
        best_row = df.loc[df['val_accuracy'].idxmax()]
        var_global_epoch = int(best_row['epoch'])
        print(f"ℹ️ Época não especificada. Selecionada automaticamente a melhor Época: {var_global_epoch} (Acurácia: {best_row['val_accuracy']:.2f}%)")
    else:
        var_global_epoch = int(epoch)

    # Buscar dados da época selecionada no dataframe
    selected_rows = df[df['epoch'] == var_global_epoch]
    if selected_rows.empty:
        print(f"⚠️ Aviso: Época {var_global_epoch} não encontrada no CSV. Usando a última época registrada.")
        var_global_epoch = int(df['epoch'].iloc[-1])
        selected_row = df.iloc[-1]
    else:
        selected_row = selected_rows.iloc[0]

    # Ajuste do estilo visual (compatibilidade com diferentes versões do Matplotlib/Seaborn)
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        plt.style.use('ggplot')

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    METRIC = df['val_accuracy']

    # --- 1. Total Loss + Mask Loss + Model Loss ---
    axes[0].plot(df['epoch'], df['total_loss'], 'o-', color='#e74c3c', linewidth=2, markersize=5, label='Total Loss')
    axes[0].plot(df['epoch'], df['mask_loss'],  'o-', color='#3498db', linewidth=2, markersize=5, label='Mask Loss (Pixels Ativos %)')
    axes[0].plot(df['epoch'], df['model_loss'], 'o-', color='#2ecc71', linewidth=2, markersize=5, label='Model Loss')

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Losses x Epoch', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # --- 2. Accuracy ---
    axes[1].plot(df['epoch'], METRIC, 'o-', color='#2ecc71', linewidth=2, markersize=5, label='Val Accuracy')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Validation Accuracy x Epoch', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Opcional: Adicionar twin axis para mostrar a porcentagem de pixels ativos junto com a acurácia
    ax2_twin = axes[1].twinx()
    ax2_twin.plot(df['epoch'], df['mask_loss'] * 100, '--', color='#3498db', alpha=0.6, linewidth=1.5, label='Active Pixels (%)')
    ax2_twin.set_ylabel('Active Pixels (%)', fontsize=10, color='#3498db')
    ax2_twin.tick_params(axis='y', labelcolor='#3498db')
    ax2_twin.grid(False)

    # --- 3. Lambda ---
    axes[2].plot(df['epoch'], df['lambda_value'], 'o-', color='#3498db', linewidth=2, markersize=5, label='Lambda (λ)')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Lambda (λ)', fontsize=12)
    axes[2].set_title('Regularization Parameter (λ)', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    # Linha vertical em todos os subplots na época selecionada
    for ax in axes:
        ax.axvline(x=var_global_epoch, color='black', linestyle='--', alpha=0.7, label=f'Selected Epoch ({var_global_epoch})')

    plt.suptitle(f"Análise de Treinamento — Época Selecionada: {var_global_epoch}", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 Gráfico salvo em: {save_path}")

    if show_plot:
        plt.show()

    # Exibição detalhada das métricas contidas no CSV para a época selecionada
    active_pixels_pct = selected_row['mask_loss'] * 100.0 if selected_row['mask_loss'] <= 1.0 else selected_row['mask_loss']
    
    print("\n" + "="*55)
    print(f"📊 MÉTRICAS CONSTATADAS NO CSV PARA A ÉPOCA {var_global_epoch}:")
    print("="*55)
    print(f"  • Época (Epoch)            : {int(selected_row['epoch'])}")
    print(f"  • Acurácia Validação (%)   : {selected_row['val_accuracy']:.2f}%")
    print(f"  • Pixels Ativos (Mask Loss): {active_pixels_pct:.2f}% ({selected_row['mask_loss']:.4f})")
    print(f"  • Loss do Modelo           : {selected_row['model_loss']:.4f}")
    print(f"  • Loss Total               : {selected_row['total_loss']:.4f}")
    print(f"  • Parâmetro Lambda (λ)     : {selected_row['lambda_value']:.6f}")
    if 'lambda_patience_count' in selected_row:
        print(f"  • Contagem de Paciência λ  : {int(selected_row['lambda_patience_count'])}")
    print("="*55 + "\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plota o histórico do training_log.csv para análise visual.")
    parser.add_argument("--csv_path", type=str, required=True, help="Caminho para o arquivo training_log.csv")
    parser.add_argument("--epoch", type=int, default=None, help="Época para destacar (opcional).")
    parser.add_argument("--save_path", type=str, default=None, help="Caminho para salvar a figura (ex: plot.png).")
    parser.add_argument("--no_show", action="store_true", help="Não exibe a janela do gráfico (útil para scripts de servidor).")

    args = parser.parse_args()
    plot_training_log(
        csv_path=args.csv_path,
        epoch=args.epoch,
        save_path=args.save_path,
        show_plot=not args.no_show
    )
