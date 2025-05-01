import src.FastTokenBasedReplay as ftr
from src.PetriNet import PetriNet, Marking
from src.EventLog import EventLog
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import apply as replay_fitness
import matplotlib.pyplot as plt
from pm4py.algo.discovery.inductive.algorithm import apply as pm4py_inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
import itertools
from pm4py.algo.discovery.alpha.algorithm import apply as alpha_miner
import random

def init_discovered_petri_net_and_eventlog(num_traces):
    def limited_permutations(s, X):
        count = 0
        for p in itertools.permutations(s):
            p = list(p)
            random.shuffle(p)
            p = tuple(p)
            
            if count >= X:
                break
            yield ''.join(p)
            count += 1

    # traces 
    s = "ABCDEFGHIJKLMNOPQR"
    permutations = list(limited_permutations(s, num_traces))
    eventlog = EventLog.from_trace_list(permutations)
    
    pm4py_event_log = eventlog.to_pm4py()
    #pm4py_pt = pm4py_inductive_miner(pm4py_event_log)
    #pm4py_net, init, end = pt_converter.apply(pm4py_pt, variant=pt_converter.Variants.TO_PETRI_NET)
    pm4py_net, init, end = alpha_miner(pm4py_event_log)
    our_net = PetriNet.from_pm4py(pm4py_net, init, end)
    # our_net.set_final_marking(Marking({"End": 1}))
    # our_net.set_initial_marking(Marking({"Start": 1}))
    
    return eventlog, our_net




def main():
    for _ in range(1000):
        num_traces = 100
        
        our_eventlog, our_net = init_discovered_petri_net_and_eventlog(num_traces)
        
        c_petri_net = our_net.to_fast_token_based_replay()
        c_eventlog = our_eventlog.to_fast_token_based_replay()
        
        ftr_fitness = ftr.calculate_fitness(c_eventlog, c_petri_net, False, False)
        
        pm4py_fitness = replay_fitness(our_eventlog.to_pm4py(), *our_net.to_pm4py())["log_fitness"]
        

        assert ftr_fitness == pm4py_fitness, f"FTR: {ftr_fitness}, PM4PY: {pm4py_fitness}"
    
    
if __name__ == "__main__":
    main()