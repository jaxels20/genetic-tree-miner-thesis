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
from pyinstrument import Profiler
import numpy as np
from pm4py.sim import play_out, generate_process_tree

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

        c_petri_net = petri_net.to_fast_token_based_replay()
        c_eventlog = eventlog.to_fast_token_based_replay()
        
        start = time.time()
        FastTokenBasedReplay = ftr.calculate_fitness(c_eventlog, c_petri_net, False, False)
        end = time.time()
        samples.append(end - start)
            
    return samples

def test_fast_token_based_replay_with_prefix_caching(num_traces=[]):
    samples = []
    for i in num_traces:        
        eventlog, petri_net = init_discovered_petri_net_and_eventlog(i)

        c_petri_net = petri_net.to_fast_token_based_replay()
        c_eventlog = eventlog.to_fast_token_based_replay()
        
        start = time.time()
        FastTokenBasedReplay = ftr.calculate_fitness(c_eventlog, c_petri_net, True, False)
        end = time.time()
        samples.append(end - start)
    return samples

def test_fast_token_based_replay_with_suffix_caching(num_traces=[]):
    samples = []
    for i in num_traces:
        eventlog, petri_net = init_discovered_petri_net_and_eventlog(i)

        c_petri_net = petri_net.to_fast_token_based_replay()
        c_eventlog = eventlog.to_fast_token_based_replay()
        
        start = time.time()
        FastTokenBasedReplay = ftr.calculate_fitness(c_eventlog, c_petri_net, False, True)
        end = time.time()
        samples.append(end - start)

    
    return samples

def test_fast_token_based_replay_with_prefix_and_suffix_caching(num_traces=[]):
    samples = []
    for i in num_traces:        
        eventlog, petri_net = init_discovered_petri_net_and_eventlog(i)

        c_petri_net = petri_net.to_fast_token_based_replay()
        c_eventlog = eventlog.to_fast_token_based_replay()
        
        start = time.time()
        FastTokenBasedReplay = ftr.calculate_fitness(c_eventlog, c_petri_net, True, True)
        end = time.time()
        samples.append(end - start)
        
    
    return samples

def test_pm4py_token_based_replay(num_traces=[]):
    samples = []
    for i in num_traces:
        eventlog, petri_net = init_discovered_petri_net_and_eventlog(i)
        PetriNet, initital_marking, final_marking = petri_net.to_pm4py()
        #Print the number of transitions
        EventLog = eventlog.to_pm4py()
        
        start = time.time()
        replay_fitness(EventLog, PetriNet, initital_marking, final_marking )
        end = time.time()
        
        samples.append(end - start)
        
    return samples

def test_pm4py_alignment(num_traces=[]):
    samples = []
    for i in num_traces:
        eventlog, petri_net = init_synthetic_eventlog_and_petri_net(i)
        PetriNet, initital_marking, final_marking = petri_net.to_pm4py()
        EventLog = eventlog.to_pm4py()
        
        start = time.time()
        align_petri_net(EventLog, PetriNet, initital_marking, final_marking)
        end = time.time()
        samples.append(end - start)
        
        # print the time in milliseconds
        print(f"pm4py took {(end - start) * 1000} milliseconds")
    return samples

def real_life_evaluation():
    eventlog_dir = "./real_life_datasets"
    data = {}
    for filename in os.listdir(eventlog_dir):
        if filename.endswith(".xes"):
            print(f"Processing {filename}")
            our_event_log = EventLog.load_xes(os.path.join(eventlog_dir, filename))
            pm4py_event_log = our_event_log.to_pm4py()
            pm4py_pt = pm4py_inductive_miner(pm4py_event_log)
            pm4py_net, init, end = pt_converter.apply(pm4py_pt, variant=pt_converter.Variants.TO_PETRI_NET)
            
            our_net = PetriNet.from_pm4py(pm4py_net)
            our_net.set_final_marking(Marking({"End": 1}))
            our_net.set_initial_marking(Marking({"Start": 1}))
            
            # print stats for this event log and petri net
            print(f"Number of traces: {len(our_event_log)}")
            print(f"Number of transitions: {len(our_net.transitions)}")
            print(f"Number of silent transitions: {len([t for t in our_net.transitions if t.is_silent()])}")
            print(f"Number of Unique Activities: {len(our_event_log.unique_activities())}")
            
            
            # FastTokenBasedReplay without caching
            c_petri_net = our_net.to_fast_token_based_replay()
            c_eventlog = our_event_log.to_fast_token_based_replay()
            our_start_without_caching = time.time()
            FastTokenBasedReplay = ftr.calculate_fitness(c_eventlog, c_petri_net, False, False)
            our_end_without_caching = time.time()
            
            
            # # FastTokenBasedReplay with prefix caching
            # c_petri_net = our_net.to_fast_token_based_replay()
            # c_eventlog = our_event_log.to_fast_token_based_replay()
            # our_start_with_prefix_caching = time.time()
            # FastTokenBasedReplay = ftr.calculate_fitness(c_eventlog, c_petri_net, True, False)
            # our_end_with_prefix_caching = time.time()
            
            
            # # FastTokenBasedReplay with suffix caching
            # c_petri_net = our_net.to_fast_token_based_replay()
            # c_eventlog = our_event_log.to_fast_token_based_replay()
            # our_start_with_suffix_caching = time.time()
            # FastTokenBasedReplay = ftr.calculate_fitness(c_eventlog, c_petri_net, False, True)
            # our_end_with_suffix_caching = time.time()
            
            
            
            # # FastTokenBasedReplay with prefix and suffix caching
            # c_petri_net = our_net.to_fast_token_based_replay()
            # c_eventlog = our_event_log.to_fast_token_based_replay()
            # our_start_with_prefix_and_suffix_caching = time.time()
            # FastTokenBasedReplay = ftr.calculate_fitness(c_eventlog, c_petri_net, True, True)
            # our_end_with_prefix_and_suffix_caching = time.time()
            
            
            # pm4py
            pm4py_start = time.time()
            replay_fitness(pm4py_event_log, pm4py_net, init, end)
            # with SuppressPrints():
            #     replay_fitness(pm4py_event_log, pm4py_net, init, end)
            pm4py_end = time.time()
        
            data[filename] = {"FastTokenBasedReplay (without caching)": our_end_without_caching - our_start_without_caching, 
                              "pm4py": pm4py_end - pm4py_start,
                              #"FastTokenBasedReplay (with prefix caching)": our_end_with_prefix_caching - our_start_with_prefix_caching,
                              #"FastTokenBasedReplay (with suffix caching)": our_end_with_suffix_caching - our_start_with_suffix_caching,
                              #"FastTokenBasedReplay (with prefix and suffix caching)": our_end_with_prefix_and_suffix_caching - our_start_with_prefix_and_suffix_caching
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

    # Show plot
    plt.tight_layout()
    plt.show()  
    
def synthetic_evaluation():
    num_traces = [500, 1_000, 1_500, 2_000, 2_500]
    ftr_without_caching_times = test_fast_token_based_replay_without_caching(num_traces)
    ftr_with_prefix_caching_times = test_fast_token_based_replay_with_prefix_caching(num_traces)
    ftr_with_suffix_caching_times = test_fast_token_based_replay_with_suffix_caching(num_traces)
    ftr_with_prefix_and_suffix_times = test_fast_token_based_replay_with_prefix_and_suffix_caching(num_traces)
    pm4py_token_based_times = test_pm4py_token_based_replay(num_traces)
    
    # convert to ms and round to 2 decimal places
    
    print(f"FastTokenBasedReplay without caching: {[round(x * 1000, 2) for x in ftr_without_caching_times]}")
    print(f"FastTokenBasedReplay with prefix caching: {[round(x * 1000, 2) for x in ftr_with_prefix_caching_times]}")
    print(f"FastTokenBasedReplay with suffix caching: {[round(x * 1000, 2) for x in ftr_with_suffix_caching_times]}")
    print(f"FastTokenBasedReplay with prefix and suffix caching: {[round(x * 1000, 2) for x in ftr_with_prefix_and_suffix_times]}")
    print(f"pm4py token based: {[round(x * 1000, 2) for x in pm4py_token_based_times]}")
    
    
    #pm4py_alignment_times = test_pm4py_alignment(num_traces)
    
    plt.plot(num_traces, ftr_without_caching_times, label="FastTokenBasedReplay without caching")
    plt.plot(num_traces, ftr_with_prefix_caching_times, label="FastTokenBasedReplay with prefix caching")
    plt.plot(num_traces, ftr_with_suffix_caching_times, label="FastTokenBasedReplay with suffix caching")
    plt.plot(num_traces, ftr_with_prefix_and_suffix_times, label="FastTokenBasedReplay with prefix and suffix caching")
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
    
    
    