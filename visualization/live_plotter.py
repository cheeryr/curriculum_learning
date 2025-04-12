import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

class LiveTrainingPlot:
    """Reusable plotter for real-time monitoring"""
    
    def __init__(self, 
                 update_freq: int = 500,
                 smoothing_window: int = 100):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.update_freq = update_freq
        self.smoothing_window = smoothing_window
        plt.ion()
    
    def update(self, 
               rewards: List[float], 
               difficulties: Optional[List[float]] = None,
               stage_changes: Optional[List[int]] = None):
        """Update plots with new data"""
        # Clear and redraw rewards
        self.ax1.clear()
        self.ax1.plot(rewards, 'b-', alpha=0.3)
        smooth = np.convolve(rewards, np.ones(self.smoothing_window)/self.smoothing_window, 'valid')
        self.ax1.plot(smooth, 'b-')
        
        # Add stage markers if provided
        if stage_changes:
            for x in stage_changes:
                self.ax1.axvline(x, color='g', linestyle='--', alpha=0.5)
        
        # Clear and redraw difficulty if available
        if difficulties:
            self.ax2.clear()
            self.ax2.plot(difficulties, 'r-')
            if stage_changes:
                for x in stage_changes:
                    self.ax2.axvline(x, color='g', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    def save(self, path: str):
        """Save current figure"""
        self.fig.savefig(path)
        plt.close()