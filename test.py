# THIS FILE TESTS FASTTOKENBASEDREPLAY AND COMPARES THE OUTPUTS TO PM4PY
from src.EventLog import EventLog
from src.PetriNet import PetriNet, Marking
import src.FastTokenBasedReplay as FastTokenBasedReplay
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import apply as replay_fitness

def test_simple_sequence():
    eventlog = EventLog.from_trace_list(["ABC"])
    petrinet = PetriNet()
    petrinet.add_transition("A")
    petrinet.add_transition("B")
    petrinet.add_transition("C")
    
    petrinet.add_place("Start")
    petrinet.add_place("End")
    petrinet.add_place("A->B")
    petrinet.add_place("B->C")
    
    petrinet.add_arc("Start", "A")
    petrinet.add_arc("A", "A->B")
    petrinet.add_arc("A->B", "B")
    petrinet.add_arc("B", "B->C")
    petrinet.add_arc("B->C", "C")
    petrinet.add_arc("C", "End")
    
    petrinet.set_initial_marking(Marking({"Start": 1}))
    petrinet.set_final_marking(Marking({"End": 1}))
    
    ftr_fitness, ftr_precision = FastTokenBasedReplay.calculate_fitness_and_precision(eventlog.to_fast_token_based_replay(), petrinet.to_fast_token_based_replay())
    
    pm4py_fitness = replay_fitness(eventlog.to_pm4py(), *petrinet.to_pm4py())["log_fitness"]
    #pm4py_precision = objective.precision()
    print(f"FastTokenBasedReplay fitness: {ftr_fitness}")
    print(f"PM4Py fitness: {pm4py_fitness}")
    assert ftr_fitness == pm4py_fitness

def test_simple_sequence_not_perfect():
    eventlog = EventLog.from_trace_list(["ABC", "AB"])
    petrinet = PetriNet()
    petrinet.add_transition("A")
    petrinet.add_transition("B")
    petrinet.add_transition("C")
    
    petrinet.add_place("Start")
    petrinet.add_place("End")
    petrinet.add_place("A->B")
    petrinet.add_place("B->C")
    
    petrinet.add_arc("Start", "A")
    petrinet.add_arc("A", "A->B")
    petrinet.add_arc("A->B", "B")
    petrinet.add_arc("B", "B->C")
    petrinet.add_arc("B->C", "C")
    petrinet.add_arc("C", "End")
    
    petrinet.set_initial_marking(Marking({"Start": 1}))
    petrinet.set_final_marking(Marking({"End": 1}))
    
    ftr_fitness, ftr_precision = FastTokenBasedReplay.calculate_fitness_and_precision(eventlog.to_fast_token_based_replay(), petrinet.to_fast_token_based_replay())
    
    pm4py_fitness = replay_fitness(eventlog.to_pm4py(), *petrinet.to_pm4py())["log_fitness"]
    #pm4py_precision = objective.precision()
    print(f"FastTokenBasedReplay fitness: {ftr_fitness}")
    print(f"PM4Py fitness: {pm4py_fitness}")
    assert ftr_fitness == pm4py_fitness

def test_simple_loop():
    eventlog = EventLog.from_trace_list(["ABBC", "ABBBC", "ABBBBC"])
    petrinet = PetriNet()
    petrinet.empty()
    petrinet.add_transition("A")
    petrinet.add_transition("B")
    petrinet.add_transition("C")
    
    petrinet.add_place("Start")
    petrinet.add_place("End")
    petrinet.add_place("p1")

    
    petrinet.add_arc("Start", "A")
    petrinet.add_arc("A", "p1")
    petrinet.add_arc("p1", "B")
    petrinet.add_arc("B", "p1")
    petrinet.add_arc("p1", "C")
    petrinet.add_arc("C", "End")
    
    petrinet.set_initial_marking(Marking({"Start": 1}))
    petrinet.set_final_marking(Marking({"End": 1}))
    
    ftr_fitness, ftr_precision = FastTokenBasedReplay.calculate_fitness_and_precision(eventlog.to_fast_token_based_replay(), petrinet.to_fast_token_based_replay())
    
    pm4py_fitness = replay_fitness(eventlog.to_pm4py(), *petrinet.to_pm4py())["log_fitness"]
    #pm4py_precision = objective.precision()
    print(f"FastTokenBasedReplay fitness: {ftr_fitness}")
    print(f"PM4Py fitness: {pm4py_fitness}")
    assert ftr_fitness == pm4py_fitness

def test_simple_loop_not_perfect():
    eventlog = EventLog.from_trace_list(["AB"])
    petrinet = PetriNet()
    petrinet.empty()
    petrinet.add_transition("A")
    petrinet.add_transition("B")
    petrinet.add_transition("C")
    
    petrinet.add_place("Start")
    petrinet.add_place("End")
    petrinet.add_place("p1")

    
    petrinet.add_arc("Start", "A")
    petrinet.add_arc("A", "p1")
    petrinet.add_arc("p1", "B")
    petrinet.add_arc("B", "p1")
    petrinet.add_arc("p1", "C")
    petrinet.add_arc("C", "End")
    
    petrinet.set_initial_marking(Marking({"Start": 1}))
    petrinet.set_final_marking(Marking({"End": 1}))
    
    ftr_fitness, ftr_precision = FastTokenBasedReplay.calculate_fitness_and_precision(eventlog.to_fast_token_based_replay(), petrinet.to_fast_token_based_replay())
    pm4py_fitness = replay_fitness(eventlog.to_pm4py(), *petrinet.to_pm4py())["log_fitness"]
    #pm4py_precision = objective.precision()
    print(f"FastTokenBasedReplay fitness: {ftr_fitness}")
    print(f"PM4Py fitness: {pm4py_fitness}")
    assert ftr_fitness == pm4py_fitness

def test_simple_silent_transition():
    eventlog = EventLog.from_trace_list(["B"])
    petrinet = PetriNet()
    petrinet.empty()
    petrinet.add_transition("A")
    petrinet.add_transition("B")
    petrinet.add_transition("tau_1")
    
    petrinet.add_place("Start")
    petrinet.add_place("End")
    petrinet.add_place("p1")

    
    petrinet.add_arc("Start", "A")
    petrinet.add_arc("Start", "tau_1")
    petrinet.add_arc("A", "p1")
    petrinet.add_arc("tau_1", "p1")
    petrinet.add_arc("p1", "B")
    petrinet.add_arc("B", "End")
    
    petrinet.set_initial_marking(Marking({"Start": 1}))
    petrinet.set_final_marking(Marking({"End": 1}))
    
    print(f"_______________FTR_______")
    ftr_fitness, ftr_precision = FastTokenBasedReplay.calculate_fitness_and_precision(eventlog.to_fast_token_based_replay(), petrinet.to_fast_token_based_replay())
    
    print(f"_______________PM4PY_______")
    pm4py_fitness = replay_fitness(eventlog.to_pm4py(), *petrinet.to_pm4py())["log_fitness"]
    #pm4py_precision = objective.precision()
    print(f"FastTokenBasedReplay fitness: {ftr_fitness}")
    print(f"PM4Py fitness: {pm4py_fitness}")
    assert ftr_fitness == pm4py_fitness


if __name__ == "__main__":
    test_simple_sequence()
    test_simple_loop()
    test_simple_loop_not_perfect()
    test_simple_silent_transition()



