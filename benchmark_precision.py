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
from pm4py.conformance import precision_token_based_replay 
from src.EventLog import EventLog
from src.Filtering import Filtering
import numpy as np

def time_fast_token_based_replay(eventlog, petri_net):
    c_petri_net = petri_net.to_fast_token_based_replay()
    c_eventlog = eventlog.to_fast_token_based_replay()
    
    start = time.time()
    FastTokenBasedReplay = ftr.calculate_precision(c_eventlog, c_petri_net)
    end = time.time()
    
    return end - start

def time_pm4py(eventlog: EventLog, petrinet: PetriNet):
    pm4py_pn, init, final = petrinet.to_pm4py()
    pm4py_el = eventlog.to_pm4py()
    start = time.time()
    fitness = precision_token_based_replay(pm4py_el, pm4py_pn, init, final)
    end = time.time()
    
    return end - start

def real_life_evaluation():
    eventlog_dir = "./real_life_datasets/"
    data = {}
    for folder in os.listdir(eventlog_dir):
        if not os.path.isdir(eventlog_dir + folder):
            continue
        for filename in os.listdir(eventlog_dir + folder):
            if filename.endswith(".xes"): #and filename == "BPI_Challenge_2017.xes":
                print(f"Processing {filename}")
                if "2015" in filename:
                    continue
                our_event_log = EventLog.load_xes(os.path.join(eventlog_dir, folder, filename))
                
                # filter the traces
                our_event_log = Filtering.filter_eventlog_by_top_percentage_unique(our_event_log, 0.1, include_all_activities=False)
                
                pm4py_event_log = our_event_log.to_pm4py()
                pm4py_pt = pm4py_inductive_miner(pm4py_event_log)
                pm4py_net, init, end = pt_converter.apply(pm4py_pt, variant=pt_converter.Variants.TO_PETRI_NET)
                
                our_net = PetriNet.from_pm4py(pm4py_net, init, end)
                
                data[filename] = {"FastTokenBasedReplay": time_fast_token_based_replay(our_event_log, our_net),
                                #"pm4py": time_pm4py(our_event_log, our_net),
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
    # ax.set_yscale('log')

    # Show plot
    plt.tight_layout()
    fig.savefig('benchmark_results/precision_comparison.png')  
    
if __name__ == "__main__":
    real_life_evaluation()