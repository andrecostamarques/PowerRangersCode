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