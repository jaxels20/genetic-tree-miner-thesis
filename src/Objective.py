from src.ProcessTree import ProcessTree, Operator
from src.EventLog import EventLog
from src.ProcessTreeRegister import ProcessTreeRegister
from src.SupressPrints import SuppressPrints
from src.Population import Population
from src.PetriNet import PetriNet
import src.FastTokenBasedReplay as FastTokenBasedReplay

from typing import Union
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import apply as replay_fitness
from pm4py.algo.evaluation.precision.variants.etconformance_token import apply as precision
from pm4py.algo.evaluation.generalization.variants.token_based import apply as generalization
from pm4py.algo.evaluation.simplicity.variants.arc_degree import apply as simplicity



class Objective:
    def __init__(self, event_log: EventLog):        
        # Initialize eventlog, pm4py_eventlog, and ftr_eventlog    
        self.eventlog = event_log
        self.event_log_pm4py = event_log.to_pm4py()
        self.ftr_eventlog = self.eventlog.to_fast_token_based_replay()
        
    def simplicity(self, pm4py_pn):
        with SuppressPrints():
            simplicity_value = simplicity(pm4py_pn)
        return simplicity_value
    
    def refined_simplicity(self, pm4py_pn):
        max_places = 100
        simplicity = len(pm4py_pn.get_places()) / max_places
        simplicity = 1 - simplicity
        simplicity = max(0, simplicity)
        return simplicity 
    
    def generalization(self, pm4py_pn, initial_marking, final_marking):
        with SuppressPrints():
            generalization_value = generalization(self.event_log_pm4py, pm4py_pn, initial_marking, final_marking)
        return generalization_value
    
    def average_trace_fitness(self, pm4py_pn, inital_marking, final_marking):
        with SuppressPrints():
            fitness = replay_fitness(self.event_log_pm4py, pm4py_pn, inital_marking, final_marking)
        return fitness['average_trace_fitness']

    def perc_fit_traces(self, pm4py_pn, inital_marking, final_marking):
        with SuppressPrints():
            fitness = replay_fitness(self.event_log_pm4py, pm4py_pn, inital_marking, final_marking)
        return fitness['perc_fit_traces']
        
    def log_fitness(self, pm4py_pn, inital_marking, final_marking):
        with SuppressPrints():
            fitness = replay_fitness(self.event_log_pm4py, pm4py_pn, inital_marking, final_marking)
        return fitness['log_fitness']
          
    def percentage_of_fitting(self, pm4py_pn, inital_marking, final_marking):
        with SuppressPrints():
            fitness = replay_fitness(self.event_log_pm4py, pm4py_pn, inital_marking, final_marking)
        return fitness['perc_fit_traces']

    def precision(self, pm4py_pn, inital_marking, final_marking):
        with SuppressPrints():
            precision_value = precision(self.event_log_pm4py, pm4py_pn, inital_marking, final_marking)
        return precision_value
    
    def ftr_fitness(self, ftr_petri_net):
        fitness = FastTokenBasedReplay.calculate_fitness(self.ftr_eventlog, ftr_petri_net, False, False)
        return fitness
    
    def ftr_precision(self, ftr_petri_net):
        precision = FastTokenBasedReplay.calculate_precision(self.ftr_eventlog, ftr_petri_net)
        return precision
    
    def fitness(self, process_tree: ProcessTree) -> float:
        weights = {
            "simplicity": 10,
            "refined_simplicity": 10,
            "average_trace_fitness": 80,
            "precision": 50,
        }
        
        pm4py_pn, initial_marking, final_marking = process_tree.to_pm4py_pn()
        ftr_pn = PetriNet.from_pm4py(pm4py_pn, initial_marking, final_marking).to_fast_token_based_replay()
        scores = {
            "simplicity": self.simplicity(pm4py_pn),
            "refined_simplicity": self.refined_simplicity(pm4py_pn),
            "average_trace_fitness": self.ftr_fitness(ftr_pn),
            "precision": self.ftr_precision(ftr_pn)
        }
        return sum(scores[key] * weights[key] for key in scores.keys())
    
    def evaluate_population(self, population: Population):
        for tree in population.trees:
            tree.fitness = self.fitness(tree)
    
