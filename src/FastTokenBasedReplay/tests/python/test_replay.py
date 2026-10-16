import pytest

import FastTokenBasedReplay as ftr


def sequence_net():
    """start -> A -> p1 -> B -> end"""
    net = ftr.PetriNet()
    for p in ["start", "p1", "end"]:
        net.add_place(ftr.Place(p, 0))
    for t in ["A", "B"]:
        net.add_transition(ftr.Transition(t))
    for source, target in [("start", "A"), ("A", "p1"), ("p1", "B"), ("B", "end")]:
        net.add_arc(ftr.Arc(source, target, 1))
    initial, final = ftr.Marking(), ftr.Marking()
    initial.add_place("start", 1)
    final.add_place("end", 1)
    net.set_initial_marking(initial)
    net.set_final_marking(final)
    return net


def flower_net():
    """start -> tau_0 -> p (A and B loop on p) -> tau_1 -> end"""
    net = ftr.PetriNet()
    for p in ["start", "p", "end"]:
        net.add_place(ftr.Place(p, 0))
    for t in ["A", "B", "tau_0", "tau_1"]:
        net.add_transition(ftr.Transition(t))
    arcs = [("start", "tau_0"), ("tau_0", "p"), ("p", "A"), ("A", "p"), ("p", "B"), ("B", "p"), ("p", "tau_1"), ("tau_1", "end")]
    for source, target in arcs:
        net.add_arc(ftr.Arc(source, target, 1))
    initial, final = ftr.Marking(), ftr.Marking()
    initial.add_place("start", 1)
    final.add_place("end", 1)
    net.set_initial_marking(initial)
    net.set_final_marking(final)
    return net


def test_fitting_trace():
    log = ftr.from_traces([["A", "B"]])
    assert ftr.calculate_fitness(log, sequence_net()) == pytest.approx(1.0)
    assert ftr.calculate_precision(log, sequence_net()) == pytest.approx(1.0)


def test_non_fitting_trace_lowers_fitness():
    log = ftr.from_traces([["B", "A"]])
    assert ftr.calculate_fitness(log, sequence_net()) < 1.0


def test_silent_transitions_are_replayed():
    log = ftr.from_traces([["A", "B", "A"], ["B"]])
    assert ftr.calculate_fitness(log, flower_net()) == pytest.approx(1.0)


def test_flower_model_is_less_precise():
    log = ftr.from_traces([["A", "B"]])
    assert ftr.calculate_precision(log, flower_net()) < ftr.calculate_precision(log, sequence_net())


def test_keyword_arguments_and_caching_variants():
    log = ftr.from_traces([["A", "B"], ["A", "B"], ["B", "A"]])
    expected = ftr.calculate_fitness(log=log, net=sequence_net())
    assert ftr.calculate_fitness(log, sequence_net(), prefix_caching=True) == pytest.approx(expected)
    assert ftr.calculate_fitness(log, sequence_net(), suffix_caching=True) == pytest.approx(expected)


def test_combined_caching_is_rejected():
    log = ftr.from_traces([["A", "B"]])
    with pytest.raises(ValueError):
        ftr.calculate_fitness(log, sequence_net(), prefix_caching=True, suffix_caching=True)


# Minimal stand-ins with the same attributes as pm4py's Petri net objects
class Node:
    def __init__(self, name, label=None):
        self.name = name
        self.label = label


class FakeArc:
    def __init__(self, source, target, weight=1):
        self.source, self.target, self.weight = source, target, weight


class FakeNet:
    def __init__(self, places, transitions, arcs):
        self.places, self.transitions, self.arcs = places, transitions, arcs


def fake_sequence_net(labels=("A", "B"), place_names=("source", "p_1", "sink")):
    places = [Node(n) for n in place_names]
    transitions = [Node(f"t{i}", label) for i, label in enumerate(labels)]
    arcs = [
        FakeArc(places[0], transitions[0]),
        FakeArc(transitions[0], places[1]),
        FakeArc(places[1], transitions[1]),
        FakeArc(transitions[1], places[2]),
    ]
    return FakeNet(places, transitions, arcs), {places[0]: 1}, {places[2]: 1}


def test_from_pm4py_net():
    net, im, fm = fake_sequence_net()
    log = ftr.from_traces([["A", "B"]])
    assert ftr.calculate_fitness(log, ftr.from_pm4py_net(net, im, fm)) == pytest.approx(1.0)
    # The input net is left untouched
    assert [t.name for t in net.transitions] == ["t0", "t1"]


def test_from_pm4py_net_silent_transition():
    net, im, fm = fake_sequence_net(labels=("A", None))
    log = ftr.from_traces([["A"]])
    assert ftr.calculate_fitness(log, ftr.from_pm4py_net(net, im, fm)) == pytest.approx(1.0)


def test_from_pm4py_net_renames_clashing_places():
    net, im, fm = fake_sequence_net(place_names=("A", "B", "B"))
    log = ftr.from_traces([["A", "B"]])
    assert ftr.calculate_fitness(log, ftr.from_pm4py_net(net, im, fm)) == pytest.approx(1.0)


@pytest.mark.parametrize("labels", [("A", "A"), ("A", "tau_x")])
def test_from_pm4py_net_rejects_ambiguous_labels(labels):
    net, im, fm = fake_sequence_net(labels=labels)
    with pytest.raises(ValueError):
        ftr.from_pm4py_net(net, im, fm)


def test_from_pm4py_log_dataframe_orders_by_timestamp():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "case:concept:name": ["1", "1", "2", "2"],
        "concept:name": ["B", "A", "A", "B"],
        "time:timestamp": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-02"]),
    })
    log = ftr.from_pm4py_log(df)
    assert ftr.calculate_fitness(log, sequence_net()) == pytest.approx(1.0)


def test_matches_pm4py_token_replay():
    pm4py = pytest.importorskip("pm4py")
    from pm4py.algo.evaluation.precision.variants.etconformance_token import apply as pm4py_precision
    from pm4py.algo.evaluation.replay_fitness.variants.token_replay import apply as pm4py_fitness

    traces = [["a", "b", "c", "d"], ["a", "c", "b", "d"], ["a", "e", "d"], ["a", "b", "c", "b", "c", "d"]]
    df = pd_log(traces)
    net, im, fm = pm4py.discover_petri_net_inductive(df)
    eval_df = pd_log(traces + [["a", "d"], ["a", "b", "d"], ["a", "e", "c", "d"]])

    ftr_log, ftr_net = ftr.from_pm4py_log(eval_df), ftr.from_pm4py_net(net, im, fm)
    pm4py_log = pm4py.convert_to_event_log(eval_df)
    assert ftr.calculate_fitness(ftr_log, ftr_net) == pytest.approx(pm4py_fitness(pm4py_log, net, im, fm)["log_fitness"])
    assert ftr.calculate_precision(ftr_log, ftr_net) == pytest.approx(pm4py_precision(pm4py_log, net, im, fm))


def pd_log(traces):
    import pandas as pd

    rows = []
    for case, trace in enumerate(traces):
        for i, activity in enumerate(trace):
            rows.append({"case:concept:name": str(case), "concept:name": activity, "time:timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i)})
    return pd.DataFrame(rows)
