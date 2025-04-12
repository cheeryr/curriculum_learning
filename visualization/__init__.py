"""
Curriculum Learning Visualization Package

Exports:
- LiveTrainingPlot: Real-time training visualizer
- TrainingAnalyzer: Post-hoc analysis tools
- plot_comparison: Compare multiple runs
"""

from .live_plotter import LiveTrainingPlot
from .post_analysis import (TrainingAnalyzer, 
                          plot_comparison,
                          save_animation)

__all__ = [
    'LiveTrainingPlot',
    'TrainingAnalyzer',
    'plot_comparison',
    'save_animation'
]