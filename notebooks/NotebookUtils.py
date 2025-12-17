"""
Notebook Utility Module.

This module provides helper functions for Jupyter Notebooks, specifically for
visualizing training progress through animations and loading checkpoint data
iteratively.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class NotebookUtils():
    def __init__(self):
        pass
    
    def animate_notebook(frames, interval=50, cmap='viridis', figsize=(8, 6), 
                        colorbar=True, title_prefix='Frame', vmin=0, vmax=1):
        """
        Creates an HTML animation from a sequence of 2D frames.

        Useful for visualizing the evolution of masks or weights over epochs within
        a Jupyter Notebook environment.

        Args:
            frames (list of numpy.ndarray): A list of 2D arrays representing the frames.
            interval (int, optional): Delay between frames in milliseconds. Defaults to 50.
            cmap (str, optional): Matplotlib colormap name. Defaults to 'viridis'.
            figsize (tuple, optional): Figure size (width, height). Defaults to (8, 6).
            colorbar (bool, optional): Whether to display a colorbar. Defaults to True.
            title_prefix (str, optional): Prefix for the title of each frame. Defaults to 'Frame'.
            vmin (float, optional): Minimum value for colormap scaling. Defaults to 0.
            vmax (float, optional): Maximum value for colormap scaling. Defaults to 1.

        Returns:
            IPython.display.HTML: An HTML object containing the Javascript-based animation.
        """
        
        # Configurar figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plotar primeiro frame
        im = ax.imshow(frames[0], cmap=cmap, animated=True, 
                    vmin=vmin, vmax=vmax, origin='lower')
        
        if colorbar:
            plt.colorbar(im, ax=ax)
        
        ax.grid(False)
        ax.set_title(f'{title_prefix} 0')
        
        # Função de atualização
        def update(frame_idx):
            im.set_array(frames[frame_idx])
            ax.set_title(f'{title_prefix} {frame_idx}')
            return [im]
        
        # Criar animação
        ani = FuncAnimation(fig, update, frames=len(frames), 
                        interval=interval, blit=True, repeat=True)
        
        plt.close(fig)  # Evita plotar a figura estática
        
        # Retornar HTML
        return HTML(ani.to_jshtml())

    def load_checkpoints(training_id):
        """
        Iteratively loads training checkpoints for a given experiment ID.

        This generator searches for checkpoint files following the pattern
        'checkpoint_epoch_{i}.pt' in the standard checkpoint directory structure
        and yields their state dictionaries.

        Args:
            training_id (str): The unique identifier of the training session (folder name).

        Yields:
            dict: The state dictionary loaded from the checkpoint file.
        """
        i = 0
        while True:
            i += 1
            # Assumindo que você está em 'src/scripts' e os checkpoints estão em '../../checkpoints/'
            checkpoint_path = os.path.abspath(os.path.join(
                os.getcwd(), '..', '..', 'checkpoints', training_id, f'checkpoint_epoch_{i}.pt'
            ))
            
            if not os.path.exists(checkpoint_path):
                if i == 1:
                    print(f"Nenhum checkpoint encontrado para o ID '{training_id}'. Verifique o caminho.")
                break
                
            try:
                # CORREÇÃO ESSENCIAL: Permite carregar dados do NumPy e evitar o UnpicklingError
                state_dict = torch.load(
                    checkpoint_path, 
                    map_location='cpu',
                    weights_only=False 
                )
                yield state_dict
                
            except Exception as e:
                print(f"Erro ao carregar o checkpoint da época {i} em {checkpoint_path}: {e}")
                break


    def plot_checkpoint_stats(training_id, epoch_idx):
        # 1. Tentar carregar o checkpoint específico
        # (Ajuste o caminho conforme sua estrutura: ../../checkpoints/ID/...)
        path = f"../../checkpoints/{training_id}/checkpoint_epoch_{epoch_idx}.pt"
        
        try:
            check = torch.load(path, map_location='cpu', weights_only=False)
        except FileNotFoundError:
            print(f"❌ Erro: Checkpoint da época {epoch_idx} não encontrado em: {path}")
            return

        cm = np.array(check['cm'])
        
        # --- CÁLCULO DAS MÉTRICAS ---
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        eps = 1e-9
        
        acc = np.sum(tp) / np.sum(cm)
        prec = np.mean(tp / (tp + fp + eps))
        rec = np.mean(tp / (tp + fn + eps))
        f1 = 2 * (prec * rec) / (prec + rec + eps)
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [acc * 100, prec * 100, rec * 100, f1 * 100]

        # --- PLOTAGEM ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.2, 1]})
        fig.suptitle(f"Inspeção Detalhada - Época {epoch_idx} ({training_id})", fontsize=16, fontweight='bold')

        # Painel 1: Heatmap da Matriz de Confusão
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
        axes[0].set_title("Matriz de Confusão", fontweight='bold')
        axes[0].set_xlabel("Predição")
        axes[0].set_ylabel("Real")

        # Painel 2: Histograma (Barra) de Métricas
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#9b59b6']
        bars = axes[1].bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Adiciona os valores no topo das barras
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

        axes[1].set_title("Métricas Globais (Macro)", fontweight='bold')
        axes[1].set_ylim(0, 115) # Espaço extra para o texto acima da barra
        axes[1].set_ylabel("Porcentagem (%)")
        axes[1].grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
