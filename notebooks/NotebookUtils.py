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
import seaborn as sns # Adicionado para o plot_checkpoint_stats
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import pandas as pd

class NotebookUtils():
    def __init__(self):
        pass
    
    def animate_notebook(self, frames, interval, cmap='viridis', figsize=(8, 6), 
                        colorbar=True, title_prefix='Frame', vmin=0, vmax=1):
        """
        Gera uma animação HTML para o Jupyter. 
        Suporta máscaras 2D (Grayscale) e 3D (RGB - Canais, Altura, Largura).
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Função interna para tratar o formato do Tensor/Numpy
        def prepare_frame(f):
            # Se for (C, H, W), converte para (H, W, C) para o Matplotlib
            if f.ndim == 3:
                if f.shape[0] in [1, 3]: # Se os canais estiverem na frente
                    f = np.transpose(f, (1, 2, 0))
                # Se for 1 canal após o transpose, remove a dimensão extra
                if f.shape[-1] == 1:
                    f = f.squeeze(-1)
            return f

        first_frame = prepare_frame(frames[0])
        
        # Detecta se é RGB para desativar colorbar e vmin/vmax (que causariam erro)
        is_rgb = (first_frame.ndim == 3 and first_frame.shape[-1] == 3)
        
        im = ax.imshow(first_frame, 
                       cmap=None if is_rgb else cmap, 
                       animated=True, 
                       vmin=None if is_rgb else vmin, 
                       vmax=None if is_rgb else vmax, 
                       origin='lower')
        
        if colorbar and not is_rgb:
            plt.colorbar(im, ax=ax)
        
        ax.grid(False)
        ax.set_title(f'{title_prefix} 0')
        
        def update(frame_idx):
            current_frame = prepare_frame(frames[frame_idx])
            im.set_array(current_frame)
            ax.set_title(f'{title_prefix} {frame_idx}')
            return [im]
        
        ani = FuncAnimation(fig, update, frames=len(frames), 
                            interval=interval, blit=True, repeat=True)
        
        plt.close(fig)
        return HTML(ani.to_jshtml())


    def load_checkpoint(self, global_checkpoint_id, epoch, root_dir="../../checkpoints/"):
        checkpoint_path = os.path.abspath(os.path.join(
            root_dir, 
            global_checkpoint_id, 
            f'checkpoint_epoch_{epoch}.pt'
        ))
        
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: File not found at: {checkpoint_path}")
            return None
            
        try:
            state_dict = torch.load(
                checkpoint_path, 
                map_location='cpu',
                weights_only=False 
            )
            return state_dict
        except Exception as e:
            print(f"Error loading checkpoint at {checkpoint_path}: {e}")
            return None

    def load_training_log(self, global_checkpoint_id, root_dir="../../checkpoints/"):
        csv_path = os.path.abspath(os.path.join(root_dir, global_checkpoint_id, "training_log.csv"))
        if not os.path.exists(csv_path):
            print(f"❌ Erro: Log não encontrado em: {csv_path}")
            return None
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Erro ao ler CSV: {e}")
            return None

    def add_metrics_df(self, df, global_checkpoint_id, root_dir="../../checkpoints/"):
        precisions, recalls, f1_scores = [], [], []
        eps = 1e-9

        for i in range(1, len(df) + 1):
            ckpt = self.load_checkpoint(global_checkpoint_id, i, root_dir)
            if ckpt is not None and 'cm' in ckpt:
                cm = np.array(ckpt['cm'])
                tp = np.diag(cm)
                fp = np.sum(cm, axis=0) - tp
                fn = np.sum(cm, axis=1) - tp
                
                p = tp / (tp + fp + eps)
                r = tp / (tp + fn + eps)
                f1 = 2 * (p * r) / (p + r + eps)
                
                precisions.append(np.mean(p) * 100)
                recalls.append(np.mean(r) * 100)
                f1_scores.append(np.mean(f1) * 100)
            else:
                precisions.append(np.nan)
                recalls.append(np.nan)
                f1_scores.append(np.nan)

        df['val_precision'] = precisions
        df['val_recall'] = recalls
        df['val_f1_score'] = f1_scores
        return df

    # ADICIONADA: A função plot_mask que terminamos agora
    def plot_mask(self, var_global_checkpoint, epoch, figsize=(15, 5), cmap='gray'):
        ckpt = self.load_checkpoint(var_global_checkpoint, epoch)
        if ckpt is None or 'mask_state_dict' not in ckpt:
            print(f"❌ Mask not found for epoch {epoch}.")
            return None

        mask_dict = ckpt['mask_state_dict']
        m = mask_dict['mask'].squeeze().cpu().detach()
        processed_mask = torch.round(torch.sigmoid(m)).numpy()
        
        is_multichannel = processed_mask.ndim == 3
        num_channels = processed_mask.shape[0] if is_multichannel else 1
        
        fig, axes = plt.subplots(1, num_channels, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i in range(num_channels):
            ax = axes[i]
            img_data = processed_mask[i] if is_multichannel else processed_mask
            im = ax.imshow(img_data, cmap=cmap, vmin=0, vmax=1, origin='lower')
            title = f'Channel {i} - Epoch {epoch}' if is_multichannel else f'Mask - Epoch {epoch}'
            ax.set_title(title, fontweight='bold')
            ax.axis('off')
            if num_channels == 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
        return fig