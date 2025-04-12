# wrappers/__init__.py
from .stage_based import LunarLanderCurriculumWrapper  # Your original wrapper
from .adaptive import AdaptiveCurriculumWrapper        # New adaptive wrapper

__all__ = [
    'LunarLanderCurriculumWrapper',# stage based
    'AdaptiveCurriculumWrapper'#adaptive
]