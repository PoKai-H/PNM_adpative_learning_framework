"""Active learning of regime maps for pore-network (karst) simulations.

Core: loop (ActiveLearner) + surrogates + acquisition + simulator caller (simulator, runner).
Pipeline (option A): x -> karst -> labeler -> (x, label counts) -> surrogate -> acquisition.
"""
from .loop import ActiveLearner, Settings, StepResult, load_settings
from .space import Dimension, ParameterSpace
from .task import Task

__all__ = ["ActiveLearner", "Dimension", "ParameterSpace", "Settings", "StepResult", "Task", "load_settings"]
__version__ = "0.2.0"
