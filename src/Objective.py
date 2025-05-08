from src.ProcessTree import ProcessTree, Operator
from src.EventLog import EventLog
from src.SupressPrints import SuppressPrints
from src.Population import Population
from src.PetriNet import PetriNet
import src.FastTokenBasedReplay as FastTokenBasedReplay
import time
from typing import Union
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import apply as replay_fitness
from pm4py.algo.evaluation.precision.variants.etconformance_token import apply as precision
from pm4py.algo.evaluation.generalization.variants.token_based import apply as generalization
from pm4py.algo.evaluation.simplicity.variants.arc_degree import apply as simplicity


class Objective:
    """
    metric_weights: A dictionary with metric names as keys and float weights as values.
        Valid keys include:
        - 'simplicity'
        - 'refined_simplicity'
        - 'generalization'
        - 'average_trace_fitness'
        - 'log_fitness'
        - 'perc_fit_traces'
        - 'precision'
        - 'ftr_fitness'
        - 'ftr_precision'
    """
    def __init__(self, metric_weights: dict[str, float]):
        self.eventlog = None
        self.event_log_pm4py = None
        self.ftr_eventlog = None
        self.metric_weights = metric_weights

        # Dictionary mapping metric names to the actual evaluation functions
        self.metric_functions = {
            "simplicity": self.simplicity,
            "refined_simplicity": self.refined_simplicity,
            "generalization": self.generalization,
            "average_trace_fitness": self.average_trace_fitness,
            "log_fitness": self.log_fitness,
            "perc_fit_traces": self.percentage_of_fitting,
            "precision": self.precision,
            "ftr_fitness": self.ftr_fitness,
            "ftr_precision": self.ftr_precision,
        }
        
    def set_event_log(self, event_log: EventLog):
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
        try:
            fitness = FastTokenBasedReplay.calculate_fitness(self.ftr_eventlog, ftr_petri_net, False, False)
        except Exception as e:
            raise e
        return fitness
    
    def ftr_precision(self, ftr_petri_net):
        precision = FastTokenBasedReplay.calculate_precision(self.ftr_eventlog, ftr_petri_net)        
        return precision
    
    def fitness(self, process_tree: ProcessTree) -> float:        
        pm4py_pn, initial_marking, final_marking = process_tree.to_pm4py_pn()
        ftr_pn = PetriNet.from_pm4py(pm4py_pn, initial_marking, final_marking).to_fast_token_based_replay()

        total_fitness = 0.0

        for metric_name, weight in self.metric_weights.items():
            metric_func = self.metric_functions.get(metric_name)
            if not metric_func:
                raise ValueError(f"Unknown metric: {metric_name}")

            # Dynamically decide what to pass based on the metric
            if metric_name.startswith("ftr_"):
                score = metric_func(ftr_pn)
            elif metric_name in ["simplicity", "refined_simplicity"]:
                score = metric_func(pm4py_pn)
            else:
                score = metric_func(pm4py_pn, initial_marking, final_marking)

            total_fitness += weight * score

        return total_fitness
    
    def fitness_from_pn(self, pm4py_pn, init, final):
        total_fitness = 0.0
        ftr_pn = PetriNet.from_pm4py(pm4py_pn, init, final).to_fast_token_based_replay()
        for metric_name, weight in self.metric_weights.items():
            metric_func = self.metric_functions.get(metric_name)
            if not metric_func:
                raise ValueError(f"Unknown metric: {metric_name}")

            # Dynamically decide what to pass based on the metric
            if metric_name.startswith("ftr_"):
                score = metric_func(ftr_pn)
            elif metric_name in ["simplicity", "refined_simplicity"]:
                score = metric_func(pm4py_pn)
            else:
                score = metric_func(pm4py_pn, init, final)

            total_fitness += weight * score

        return total_fitness
        
        
        
    
    def evaluate_population(self, population: Population, start_time=None, time_limit=None):
        for tree in population.trees:
            
            if time_limit is not None:
                if start_time is not None and time.time() - start_time >= time_limit:
                    break
            
            if tree.fitness is None:
                tree.fitness = self.fitness(tree)
            else:
                continue
