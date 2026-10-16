"""
Converters from pm4py objects (and plain activity sequences) to the C++ replay objects.

pm4py is not imported here: the converters only rely on the attributes of pm4py's
PetriNet, Marking, EventLog and pandas DataFrame objects, so pm4py stays optional.
"""
from typing import Any, Iterable, Sequence

from ._core import Arc, Event, EventLog, Marking, PetriNet, Place, Trace, Transition

# The C++ engine treats every transition whose name starts with this prefix as silent
SILENT_PREFIX = "tau"


def from_traces(traces: Iterable[Sequence[str]]) -> EventLog:
    """
    Build an event log from activity sequences, e.g. [["a", "b"], ["a", "c"]].
    """
    log = EventLog()
    for i, activities in enumerate(traces):
        trace = Trace(f"trace_{i}", {})
        for activity in activities:
            trace.add_event(Event(str(activity), "", {}))
        log.add_trace(trace)
    return log


def from_pm4py_log(
    log: Any,
    activity_key: str = "concept:name",
    case_id_key: str = "case:concept:name",
    timestamp_key: str = "time:timestamp",
) -> EventLog:
    """
    Convert a pm4py EventLog or a pandas DataFrame in pm4py format.

    For a DataFrame, events are ordered by timestamp within each case when the
    timestamp column exists (ties keep their original order).
    """
    if hasattr(log, "groupby"):
        df = log
        if timestamp_key in df.columns:
            df = df.sort_values(timestamp_key, kind="stable")
        sequences = df.groupby(case_id_key, sort=False)[activity_key].agg(list)
        return from_traces(sequences)

    return from_traces([event[activity_key] for event in trace] for trace in log)


def from_pm4py_net(net: Any, initial_marking: Any, final_marking: Any) -> PetriNet:
    """
    Convert a pm4py Petri net with its initial and final marking.

    The C++ engine identifies transitions by name: a visible transition is named after
    its label and matched against event activities, and silent transitions (label None)
    are named tau_0, tau_1, ... The input net is not modified.

    Raises ValueError for nets the engine cannot represent faithfully: two visible
    transitions with the same label, or a visible label starting with "tau".
    """
    transition_names = {}
    visible_labels = set()
    num_silent = 0
    for t in net.transitions:
        if t.label is None:
            name = f"{SILENT_PREFIX}_{num_silent}"
            num_silent += 1
        else:
            name = t.label
            if name.startswith(SILENT_PREFIX):
                raise ValueError(f"visible transition label {name!r} starts with {SILENT_PREFIX!r} and would be treated as silent")
            if name in visible_labels:
                raise ValueError(f"duplicate visible transition label {name!r} is not supported")
            visible_labels.add(name)
        transition_names[id(t)] = name

    # Places keep their pm4py name unless it is a duplicate or clashes with a transition name
    taken = set(transition_names.values())
    place_names = {}
    for p in net.places:
        name = p.name
        while name in taken:
            name = f"_{name}"
        taken.add(name)
        place_names[id(p)] = name

    ftr_net = PetriNet()
    for p in net.places:
        ftr_net.add_place(Place(place_names[id(p)], 0))
    for t in net.transitions:
        ftr_net.add_transition(Transition(transition_names[id(t)]))

    node_names = {**place_names, **transition_names}
    for arc in net.arcs:
        ftr_net.add_arc(Arc(node_names[id(arc.source)], node_names[id(arc.target)], arc.weight))

    ftr_net.set_initial_marking(_convert_marking(initial_marking, place_names))
    ftr_net.set_final_marking(_convert_marking(final_marking, place_names))
    return ftr_net


def _convert_marking(marking: Any, place_names: dict) -> Marking:
    ftr_marking = Marking()
    for place, tokens in marking.items():
        ftr_marking.add_place(place_names[id(place)], tokens)
    return ftr_marking
