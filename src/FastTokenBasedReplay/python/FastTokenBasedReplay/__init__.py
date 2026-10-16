"""Fast token-based replay fitness and precision for Petri nets, implemented in C++."""
from importlib.metadata import PackageNotFoundError, version

from ._core import (
    Arc,
    Event,
    EventLog,
    Marking,
    PetriNet,
    Place,
    Trace,
    Transition,
    calculate_fitness,
    calculate_precision,
)
from ._convert import from_pm4py_log, from_pm4py_net, from_traces

try:
    __version__ = version("FastTokenBasedReplay")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Arc",
    "Event",
    "EventLog",
    "Marking",
    "PetriNet",
    "Place",
    "Trace",
    "Transition",
    "calculate_fitness",
    "calculate_precision",
    "from_pm4py_log",
    "from_pm4py_net",
    "from_traces",
]
