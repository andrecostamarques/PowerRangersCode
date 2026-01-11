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
import seaborn as sns # Added for plot_checkpoint_stats
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import pandas as pd

class NotebookUtils():
    """
    Utility class for Jupyter Notebook operations.

    This class provides methods to visualize training progress, load checkpoints,
    and analyze training logs within a notebook environment.
    """
    def __init__(self):
        """Initializes the NotebookUtils instance."""
        pass
    
    def animate_notebook(self, frames, ratio, interval=50, cmap='viridis', figsize=(8, 6), 
                     colorbar=True, title_prefix='Frame', vmin=0, vmax=1):
        """
        Creates an HTML animation from a list of frames.

        Args:
            frames (list): A list of frames (numpy arrays) to animate.
            ratio (int): The ratio of frames to epochs (e.g., if frames are saved every 10 epochs).
            interval (int, optional): Delay between frames in milliseconds. Defaults to 50.
            cmap (str, optional): Colormap for the image. Defaults to 'viridis'.
            figsize (tuple, optional): Figure size (width, height). Defaults to (8, 6).
            colorbar (bool, optional): Whether to display a colorbar. Defaults to True.
            title_prefix (str, optional): Prefix for the title of each frame. Defaults to 'Frame'.
            vmin (float, optional): Minimum value for colormap scaling. Defaults to 0.
            vmax (float, optional): Maximum value for colormap scaling. Defaults to 1.

        Returns:
            IPython.display.HTML: An HTML object containing the Javascript animation.
        """
        
        first_frame = np.array(frames[0])
        is_multichannel = first_frame.ndim == 3
        num_channels = first_frame.shape[0] if is_multichannel else 1

        # Adjust figure size to accommodate subplots
        if is_multichannel:
            fig_width, fig_height = figsize
            figsize = (fig_width * num_channels, fig_height)

        # Configure figure and subplots
        fig, axes = plt.subplots(1, num_channels, figsize=figsize, squeeze=False)
        axes = axes.flatten() # Ensure axes is always an iterable array
        
        # Plot the first frame in each subplot
        images = []
        for i in range(num_channels):
            ax = axes[i]
            frame_data = first_frame[i] if is_multichannel else first_frame
            im = ax.imshow(np.round(1 / (1 + np.exp(-frame_data))), cmap=cmap, animated=True, 
                        vmin=vmin, vmax=vmax, origin='lower')
            images.append(im)
            
            if colorbar:
                fig.colorbar(im, ax=ax)
            
            ax.grid(False)
            title = f'Channel {i} - {title_prefix} 0' if is_multichannel else f'{title_prefix} 0'
            ax.set_title(title)
        
        # Update function
        def update(frame_idx):
            current_frames = np.array(frames[frame_idx])
            for i in range(num_channels):
                frame_data = current_frames[i] if is_multichannel else current_frames
                images[i].set_array(frame_data)
                title = f'Channel {i} - {title_prefix} {frame_idx * ratio}' if is_multichannel else f'{title_prefix} {frame_idx * ratio}'
                axes[i].set_title(title)
            return images
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=len(frames), 
                        interval=interval, blit=True, repeat=False)
        
        plt.close(fig)  # Avoid plotting the static figure
        
        # Return HTML
        return HTML(ani.to_jshtml())


    def load_checkpoint(self, global_checkpoint_id, epoch, root_dir="../../checkpoints/"):
        """
        Loads a specific checkpoint file.

        Args:
            global_checkpoint_id (str): The unique identifier for the training run.
            epoch (int): The epoch number of the checkpoint to load.
            root_dir (str, optional): The root directory where checkpoints are stored. 
                Defaults to "../../checkpoints/".

        Returns:
            dict or None: The loaded state dictionary if successful, else None.
        """
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
        """
        Loads the training log CSV file into a pandas DataFrame.

        Args:
            global_checkpoint_id (str): The unique identifier for the training run.
            root_dir (str, optional): The root directory where checkpoints are stored. 
                Defaults to "../../checkpoints/".

        Returns:
            pandas.DataFrame or None: The training log data if found, else None.
        """
        csv_path = os.path.abspath(os.path.join(root_dir, global_checkpoint_id, "training_log.csv"))
        if not os.path.exists(csv_path):
            print(f"❌ Error: Log not found at: {csv_path}")
            return None
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return None

    def add_metrics_df(self, df, global_checkpoint_id, root_dir="../../checkpoints/"):
        """
        Calculates and adds validation metrics (Precision, Recall, F1-Score) to the dataframe.

        Iterates through checkpoints corresponding to the epochs in the dataframe,
        loads the confusion matrix, and computes the metrics.

        Args:
            df (pandas.DataFrame): The dataframe containing training logs.
            global_checkpoint_id (str): The unique identifier for the training run.
            root_dir (str, optional): The root directory where checkpoints are stored. 
                Defaults to "../../checkpoints/".

        Returns:
            pandas.DataFrame: The dataframe enriched with 'val_precision', 'val_recall', 
            and 'val_f1_score' columns.
        """
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

    def plot_mask(self, var_global_checkpoint, epoch, figsize=(15, 5), cmap='gray'):
        """
        Visualizes the mask for a specific epoch.

        Loads the checkpoint, extracts the mask, applies sigmoid and rounding,
        and plots it. Handles multi-channel masks.

        Args:
            var_global_checkpoint (str): The unique identifier for the training run.
            epoch (int): The epoch number to visualize.
            figsize (tuple, optional): Size of the figure. Defaults to (15, 5).
            cmap (str, optional): Colormap for the plot. Defaults to 'gray'.

        Returns:
            matplotlib.figure.Figure or None: The figure object if successful, else None.
        """
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
