import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import List, Dict, Optional

class TrainingAnalyzer:
    """Analyze saved training runs with multiple visualization options"""
    
    def __init__(self, log_path: str):
        """
        Args:
            log_path: Path to saved training log (PKL file)
        """
        with open(log_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.validate_data()
        
    def validate_data(self):
        """Ensure required data exists"""
        required = ['rewards', 'timesteps']
        if not all(k in self.data for k in required):
            raise ValueError(f"Log missing required keys: {required}")
        
    def plot_learning_curve(self, 
                          smoothing: int = 100,
                          show_difficulty: bool = True,
                          show_stages: bool = False):
        """
        Generate professional-quality training plot
        
        Args:
            smoothing: Moving average window size
            show_difficulty: Plot difficulty progression (if available)
            show_stages: Mark curriculum stage changes (if available)
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Smooth rewards
        smooth_rewards = np.convolve(
            self.data['rewards'],
            np.ones(smoothing)/smoothing,
            'valid'
        )
        
        # Main reward plot
        ax1.plot(self.data['timesteps'][:len(smooth_rewards)], 
                smooth_rewards,
                'b-', linewidth=2,
                label=f'Reward (MA {smoothing})')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Reward', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        # Add difficulty axis if available
        if show_difficulty and 'difficulties' in self.data:
            ax2 = ax1.twinx()
            ax2.plot(self.data['timesteps'][:len(self.data['difficulties'])],
                    self.data['difficulties'],
                    'r-', alpha=0.7,
                    label='Difficulty')
            ax2.set_ylabel('Difficulty', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
        
        # Add stage markers if available
        if show_stages and 'stage_changes' in self.data:
            for change in self.data['stage_changes']:
                ax1.axvline(change, color='g', 
                           linestyle='--', alpha=0.5)
        
        plt.title('Training Performance')
        fig.tight_layout()
        return fig

def plot_comparison(run_paths: List[str],
                   labels: Optional[List[str]] = None,
                   smoothing: int = 100):
    """
    Compare multiple training runs
    
    Args:
        run_paths: List of paths to saved runs
        labels: Optional legend labels
        smoothing: Moving average window
    """
    plt.figure(figsize=(12, 6))
    
    for i, path in enumerate(run_paths):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        smooth = np.convolve(
            data['rewards'],
            np.ones(smoothing)/smoothing,
            'valid'
        )
        
        label = labels[i] if labels else Path(path).stem
        plt.plot(data['timesteps'][:len(smooth)],
                smooth,
                label=label,
                linewidth=2)
    
    plt.xlabel('Training Steps')
    plt.ylabel(f'Reward (MA {smoothing})')
    plt.title('Curriculum Learning Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()

def save_animation(log_path: str,
                  output_file: str = 'results/training_evolution.mp4',
                  fps: int = 30):
    """
    Create training progression animation
    
    Args:
        log_path: Path to training log
        output_file: Output animation path
        fps: Frames per second
    """
    with open(log_path, 'rb') as f:
        data = pickle.load(f)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = data['timesteps']
    y = data['rewards']
    
    def update(frame):
        ax.clear()
        ax.plot(x[:frame], y[:frame], 'b-')
        ax.set_xlim(0, len(x))
        ax.set_ylim(min(y), max(y)+10)
        ax.set_title(f'Training Progress (Step {x[frame]})')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Reward')
        ax.grid(True)
        return ax,
    
    anim = FuncAnimation(fig, update, frames=len(x), 
                        interval=1000/fps, blit=True)
    Path(output_file).parent.mkdir(exist_ok=True)
    anim.save(output_file, writer='ffmpeg', fps=fps)
    plt.close()