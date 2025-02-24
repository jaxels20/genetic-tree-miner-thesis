import FastTokenBasedReplay as ftr
import time
import itertools
from PetriNet import PetriNet, Marking
from EventLog import EventLog
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import apply as replay_fitness
import matplotlib.pyplot as plt
from pm4py.algo.conformance.alignments.petri_net.algorithm import apply as align_petri_net


def init_eventlog_and_petri_net(num_traces=2):
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

# Now I want to compare the processing time between pm4py and the FastTokenBasedReplay

def test_fast_token_based_replay(num_traces=[]):
    samples = []
    for i in num_traces:        
        eventlog, petri_net = init_eventlog_and_petri_net(i)

        c_petri_net = petri_net.to_fast_token_based_replay()
        c_eventlog = eventlog.to_fast_token_based_replay()
        
        start = time.time()
        FastTokenBasedReplay = ftr.calculate_fitness_and_precision(c_eventlog, c_petri_net)
        end = time.time()
        samples.append(end - start)
        
        print(f"FastTokenBasedReplay took {(end - start) * 1000} milliseconds")
    
    return samples
           
        
def test_pm4py_token_based_replay(num_traces=[]):
    samples = []
    for i in num_traces:
        eventlog, petri_net = init_eventlog_and_petri_net(i)
        PetriNet, initital_marking, final_marking = petri_net.to_pm4py()
        EventLog = eventlog.to_pm4py()
        
        start = time.time()
        replay_fitness(EventLog, PetriNet, initital_marking, final_marking )
        end = time.time()
        
        samples.append(end - start)
        
        # print the time in milliseconds
        print(f"pm4py took {(end - start) * 1000} milliseconds")
    return samples

def test_pm4py_alignment(num_traces=[]):
    samples = []
    for i in num_traces:
        eventlog, petri_net = init_eventlog_and_petri_net(i)
        PetriNet, initital_marking, final_marking = petri_net.to_pm4py()
        EventLog = eventlog.to_pm4py()
        
        start = time.time()
        align_petri_net(EventLog, PetriNet, initital_marking, final_marking)
        end = time.time()
        samples.append(end - start)
        
        # print the time in milliseconds
        print(f"pm4py took {(end - start) * 1000} milliseconds")
    return samples

if __name__ == "__main__":
    num_traces = [1_000, 10_000, 50_000, 100_000]
    #num_traces = [1_000]
    ftr_times = test_fast_token_based_replay(num_traces)
    pm4py_token_based_times = test_pm4py_token_based_replay(num_traces)
    #pm4py_alignment_times = test_pm4py_alignment(num_traces)
    
    plt.plot(num_traces, ftr_times, label="FastTokenBasedReplay")
    plt.plot(num_traces, pm4py_token_based_times, label="pm4py token based")
    #plt.plot(num_traces, pm4py_alignment_times, label="pm4py alignment")
    plt.xlabel("Number of traces")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()
    
    
    
    