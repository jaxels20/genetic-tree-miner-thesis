import src.FastTokenBasedReplay as ftr
import time
import os
import sys
import itertools
from src.PetriNet import PetriNet, Marking
from src.EventLog import EventLog
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import apply as replay_fitness
import matplotlib.pyplot as plt
from pm4py.algo.conformance.alignments.petri_net.algorithm import apply as align_petri_net
from pm4py.algo.discovery.inductive.algorithm import apply as pm4py_inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from src.SupressPrints import SuppressPrints
# from pyinstrument import Profiler
import numpy as np
from pm4py.sim import play_out, generate_process_tree
from src.Filtering import Filtering

def time_fast_token_based_replay_without_caching(eventlog, petri_net):
    c_petri_net = petri_net.to_fast_token_based_replay()
    c_eventlog = eventlog.to_fast_token_based_replay()
    
    start = time.time()
    FastTokenBasedReplay = ftr.calculate_fitness(c_eventlog, c_petri_net, False, False)
    end = time.time()
    
    return end - start

def time_fast_token_based_replay_with_prefix_caching(eventlog, petri_net):
    c_petri_net = petri_net.to_fast_token_based_replay()
    c_eventlog = eventlog.to_fast_token_based_replay()
    
    start = time.time()
    FastTokenBasedReplay = ftr.calculate_fitness(c_eventlog, c_petri_net, True, False)
    end = time.time()
    
    return end - start

def time_fast_token_based_replay_with_suffix_caching(eventlog, petri_net):
    c_petri_net = petri_net.to_fast_token_based_replay()
    c_eventlog = eventlog.to_fast_token_based_replay()
    
    start = time.time()
    FastTokenBasedReplay = ftr.calculate_fitness(c_eventlog, c_petri_net, False, True)
    end = time.time()
    
    return end - start

def time_fast_token_based_replay_with_prefix_and_suffix_caching(eventlog, petri_net):
    c_petri_net = petri_net.to_fast_token_based_replay()
    c_eventlog = eventlog.to_fast_token_based_replay()
    
    start = time.time()
    FastTokenBasedReplay = ftr.calculate_fitness(c_eventlog, c_petri_net, True, True)
    end = time.time()
    
    return end - start

def time_pm4py_token_based_replay(eventlog, petri_net):
    PetriNet, initital_marking, final_marking = petri_net.to_pm4py()
    EventLog = eventlog.to_pm4py()
    
    start = time.time()
    replay_fitness(EventLog, PetriNet, initital_marking, final_marking)
    end = time.time()
    
    return end - start

def time_pm4py_alignment(eventlog, petri_net):
    PetriNet, initital_marking, final_marking = petri_net.to_pm4py()
    EventLog = eventlog.to_pm4py()
    
    start = time.time()
    align_petri_net(EventLog, PetriNet, initital_marking, final_marking)
    end = time.time()
    
    return end - start

def init_synthetic_eventlog_and_petri_net(num_traces=2):
    # Initialize a new Petri net
    petri_net = PetriNet()
    petri_net.empty()

    petri_net.add_transition("A")
    petri_net.add_transition("B")
    petri_net.add_transition("C")
    petri_net.add_transition("D")
    petri_net.add_transition("E")
    petri_net.add_transition("F")
    petri_net.add_transition("G")
    petri_net.add_transition("H")
    petri_net.add_transition("I")
    petri_net.add_transition("J")
    petri_net.add_transition("K")
    

    petri_net.add_place("Start", tokens=1)
    petri_net.add_place("End")
    petri_net.add_place("A->B")
    petri_net.add_place("B->C")
    petri_net.add_place("C->D")
    petri_net.add_place("D->E")
    petri_net.add_place("E->F")
    petri_net.add_place("F->G")
    petri_net.add_place("G->H")
    petri_net.add_place("H->I")
    petri_net.add_place("I->J")
    petri_net.add_place("J->K")
    
    # Add arcs to the Petri net
    petri_net.add_arc("Start", "A")
    petri_net.add_arc("A", "A->B")
    petri_net.add_arc("A->B", "B")
    petri_net.add_arc("B", "B->C")
    petri_net.add_arc("B->C", "C")
    petri_net.add_arc("C", "C->D")
    petri_net.add_arc("C->D", "D")
    petri_net.add_arc("D", "D->E")
    petri_net.add_arc("D->E", "E")
    petri_net.add_arc("E", "E->F")
    petri_net.add_arc("E->F", "F")
    petri_net.add_arc("F", "F->G")
    petri_net.add_arc("F->G", "G")
    petri_net.add_arc("G", "G->H")
    petri_net.add_arc("G->H", "H")
    petri_net.add_arc("H", "H->I")
    petri_net.add_arc("H->I", "I")
    petri_net.add_arc("I", "I->J")
    petri_net.add_arc("I->J", "J")
    petri_net.add_arc("J", "J->K")
    petri_net.add_arc("J->K", "K")
    petri_net.add_arc("K", "End")
    
    petri_net.set_final_marking(Marking({"End": 1}))
    petri_net.set_initial_marking(Marking({"Start": 1}))
    
    def limited_permutations(s, X):
        count = 0
        for p in itertools.permutations(s):
            if count >= X:
                break
            yield ''.join(p)
            count += 1

    # traces 
    s = "ABCDEFGHIJK"
    permutations = list(limited_permutations(s, num_traces))
    eventlog = EventLog.from_trace_list(permutations)

    return eventlog, petri_net

def init_discovered_petri_net_and_eventlog(num_traces):
    def limited_permutations(s, X):
        count = 0
        for p in itertools.permutations(s):
            if count >= X:
                break
            yield ''.join(p)
            count += 1

    # traces 
    s = "ABCDEFGHIJKLMNOPQR"
    permutations = list(limited_permutations(s, num_traces))
    eventlog = EventLog.from_trace_list(permutations)
    
    pm4py_event_log = eventlog.to_pm4py()
    pm4py_pt = pm4py_inductive_miner(pm4py_event_log)
    pm4py_net, init, end = pt_converter.apply(pm4py_pt, variant=pt_converter.Variants.TO_PETRI_NET)
    our_net = PetriNet.from_pm4py(pm4py_net)
    our_net.set_final_marking(Marking({"End": 1}))
    our_net.set_initial_marking(Marking({"Start": 1}))
    
    return eventlog, our_net

def test_fast_token_based_replay_without_caching(num_traces=[]):
    samples = []
    for i in num_traces:        
        eventlog, petri_net = init_discovered_petri_net_and_eventlog(i)
        samples.append(time_fast_token_based_replay_without_caching(eventlog, petri_net))
            
    return samples

def test_fast_token_based_replay_with_prefix_caching(num_traces=[]):
    samples = []
    for i in num_traces:        
        eventlog, petri_net = init_discovered_petri_net_and_eventlog(i)

        samples.append(time_fast_token_based_replay_with_prefix_caching(eventlog, petri_net))
    return samples

def test_fast_token_based_replay_with_suffix_caching(num_traces=[]):
    samples = []
    for i in num_traces:
        eventlog, petri_net = init_discovered_petri_net_and_eventlog(i)

        samples.append(time_fast_token_based_replay_with_suffix_caching(eventlog, petri_net))
    
    return samples

def test_fast_token_based_replay_with_prefix_and_suffix_caching(num_traces=[]):
    samples = []
    for i in num_traces:        
        eventlog, petri_net = init_discovered_petri_net_and_eventlog(i)

        samples.append(time_fast_token_based_replay_with_prefix_and_suffix_caching(eventlog, petri_net))        
        
    
    return samples

def test_pm4py_token_based_replay(num_traces=[]):
    samples = []
    for i in num_traces:
        eventlog, petri_net = init_discovered_petri_net_and_eventlog(i)
        samples.append(time_pm4py_token_based_replay(eventlog, petri_net))
        
    return samples

def test_pm4py_alignment(num_traces=[]):
    samples = []
    for i in num_traces:
        eventlog, petri_net = init_synthetic_eventlog_and_petri_net(i)
        samples.append(time_pm4py_alignment(eventlog, petri_net))
    return samples

def real_life_evaluation():
    eventlog_dir = "./real_life_datasets/"
    data = {}
    num_data_points = 1
    for folder in os.listdir(eventlog_dir):
        if not os.path.isdir(eventlog_dir + folder):
            continue
        for filename in os.listdir(eventlog_dir + folder):
            if filename.endswith(".xes"):
                print(f"Processing {filename}")
                our_event_log = EventLog.load_xes(os.path.join(eventlog_dir, folder, filename))
                
                # filter the traces
                our_event_log = Filtering.filter_eventlog_by_top_percentage_unique(our_event_log, 1, include_all_activities=False)
                
                pm4py_event_log = our_event_log.to_pm4py()
                pm4py_pt = pm4py_inductive_miner(pm4py_event_log)
                pm4py_net, init, end = pt_converter.apply(pm4py_pt, variant=pt_converter.Variants.TO_PETRI_NET)
                
                our_net = PetriNet.from_pm4py(pm4py_net, init, end)
            
                data[filename] = {"FastTokenBasedReplay (without caching)": time_fast_token_based_replay_without_caching(our_event_log, our_net),
                                "pm4py": time_pm4py_token_based_replay(our_event_log, our_net),
                                "FastTokenBasedReplay (with prefix caching)": time_fast_token_based_replay_with_prefix_caching(our_event_log, our_net),
                                #"FastTokenBasedReplay (with suffix caching)": time_fast_token_based_replay_with_suffix_caching(our_event_log, our_net),
                                #"FastTokenBasedReplay (with prefix and suffix caching)" : time_fast_token_based_replay_with_prefix_and_suffix_caching(our_event_log, our_net)
                                }
            
    
    
    # Extracting keys and values
    datasets = list(data.keys())  # X-axis labels (event logs)
    methods = list(next(iter(data.values())).keys())  # Different methods

    # Extracting values
    values = {method: [data[ds][method] for ds in datasets] for method in methods}

    # Set bar width
    bar_width = 0.15
    index = np.arange(len(datasets))  # X locations

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each method as a separate bar series
    for i, method in enumerate(methods):
        ax.bar(index + i * bar_width, values[method], bar_width, label=method)

    # Set labels
    ax.set_xlabel("Event Logs")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Execution Time Comparison for Different Methods")
    ax.set_xticks(index + (bar_width * len(methods) / 2))
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    
    # apply log scale to y axis
    ax.set_yscale('log')

    # Show plot
    plt.tight_layout()
    plt.show()
    fig.savefig('ftr_real_life_comparison.png')  
    
def synthetic_evaluation():
    num_traces = [1_000, 5_000, 50_000, 100_000]
    ftr_without_caching_times = test_fast_token_based_replay_without_caching(num_traces)
    ftr_with_prefix_caching_times = test_fast_token_based_replay_with_prefix_caching(num_traces)
    ftr_with_suffix_caching_times = test_fast_token_based_replay_with_suffix_caching(num_traces)
    #ftr_with_prefix_and_suffix_times = test_fast_token_based_replay_with_prefix_and_suffix_caching(num_traces)
    pm4py_token_based_times = test_pm4py_token_based_replay(num_traces)
    
    # convert to ms and round to 2 decimal places
    
    print(f"FastTokenBasedReplay without caching: {[round(x * 1000, 2) for x in ftr_without_caching_times]}")
    print(f"FastTokenBasedReplay with prefix caching: {[round(x * 1000, 2) for x in ftr_with_prefix_caching_times]}")
    #print(f"FastTokenBasedReplay with suffix caching: {[round(x * 1000, 2) for x in ftr_with_suffix_caching_times]}")
    #print(f"FastTokenBasedReplay with prefix and suffix caching: {[round(x * 1000, 2) for x in ftr_with_prefix_and_suffix_times]}")
    print(f"pm4py token based: {[round(x * 1000, 2) for x in pm4py_token_based_times]}")
    
        
    plt.plot(num_traces, ftr_without_caching_times, label="FastTokenBasedReplay without caching")
    plt.plot(num_traces, ftr_with_prefix_caching_times, label="FastTokenBasedReplay with prefix caching")
    plt.plot(num_traces, ftr_with_suffix_caching_times, label="FastTokenBasedReplay with suffix caching")
    #plt.plot(num_traces, ftr_with_prefix_and_suffix_times, label="FastTokenBasedReplay with prefix and suffix caching")
    plt.plot(num_traces, pm4py_token_based_times, label="pm4py token based")
    #plt.plot(num_traces, pm4py_alignment_times, label="pm4py alignment")
    
    plt.xlabel("Number of traces")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    
    real_life_evaluation()
    #synthetic_evaluation()

    #test_fast_token_based_replay_with_prefix_caching([1_000])
    #test_fast_token_based_replay_with_suffix_caching([1_000])
    
    
    