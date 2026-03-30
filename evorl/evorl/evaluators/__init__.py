from .evaluator import Evaluator
from .episode_collector import EpisodeCollector
from .ec_evaluator import EpisodeObsCollector

# BraxEvaluator pulls Brax → MuJoCo MJX → optional Warp (noisy prints without warp).
# Load it only when someone imports BraxEvaluator or uses multi-objective Brax eval.


def __getattr__(name: str):
    if name == "BraxEvaluator":
        from .mo_brax_evaluator import BraxEvaluator as _BraxEvaluator

        return _BraxEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Evaluator",
    "EpisodeCollector",
    "BraxEvaluator",
    "EpisodeObsCollector",
]
