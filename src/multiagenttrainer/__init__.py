"""MultiAgentTrainer: multi-source data collection and LLM training.

Usable as a CLI (``mat train …``), a Docker container, or a Python
library::

    from multiagenttrainer import TrainerConfig, Ingester, Runner, load_config
"""

from .config import AutoresearchConfig, TrainerConfig, TrainingConfig, load_config
from .ingest import Ingester
from .runner import ExperimentResult, Runner

__all__ = [
    "AutoresearchConfig",
    "ExperimentResult",
    "Ingester",
    "Runner",
    "TrainerConfig",
    "TrainingConfig",
    "load_config",
]

__version__ = "0.1.0"
