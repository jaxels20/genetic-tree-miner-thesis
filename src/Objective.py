from ProcessTree import ProcessTree
from EventLog import EventLog
from ProcessTree import Operator
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import apply as replay_fitness
from pm4py.algo.evaluation.precision.variants.etconformance_token import apply as precision
from pm4py.algo.evaluation.generalization.variants.token_based import apply as generalization
from pm4py.algo.evaluation.simplicity.variants.arc_degree import apply as simplicity
from SupressPrints import SuppressPrints
class ObjectiveBaseClass:
    def __init__(self, process_tree: ProcessTree, event_log: EventLog):
        self.process_tree = process_tree
        self.pm4py_pn, self.inital_marking, self.final_marking = process_tree.to_pm4py_pn()
        self.eventlog = event_log
        self.event_log_pm4py = event_log.to_pm4py()

    def simplicity(self):
        with SuppressPrints():
            simplicity_value = simplicity(self.pm4py_pn)
        return simplicity_value
    
    def generalization(self):
        with SuppressPrints():
            generalization_value = generalization(self.event_log_pm4py, self.pm4py_pn, self.inital_marking, self.final_marking)
        return generalization_value
    
    def average_trace_fitness(self):
        with SuppressPrints():
            fitness = replay_fitness(self.event_log_pm4py, self.pm4py_pn, self.inital_marking, self.final_marking)
        return fitness['average_trace_fitness']

    def perc_fit_traces(self):
        with SuppressPrints():
            fitness = replay_fitness(self.event_log_pm4py, self.pm4py_pn, self.inital_marking, self.final_marking)
        return fitness['perc_fit_traces']
        
    def log_fitness(self):
        with SuppressPrints():
            fitness = replay_fitness(self.event_log_pm4py, self.pm4py_pn, self.inital_marking, self.final_marking)
        return fitness['log_fitness']
          
    def percentage_of_fitting_traces(self):
        with SuppressPrints():
            fitness = replay_fitness(self.event_log_pm4py, self.pm4py_pn, self.inital_marking, self.final_marking)
        return fitness['perc_fit_traces']

    def precision(self):
        with SuppressPrints():
            precision_value = precision(self.event_log_pm4py, self.pm4py_pn, self.inital_marking, self.final_marking)
        return precision_value
    
    def fitness(self) -> float:
        raise NotImplementedError


class SimpleWeightedScore(ObjectiveBaseClass):
    def __init__(self, process_tree: ProcessTree, event_log: EventLog):
        super().__init__(process_tree, event_log)
    
    def weighted_score(self, scores: dict[str, float], weights: dict[str, float] = None) -> float:
        # Only works if the scores and weights have the same keys 
        
        if weights is None:
            weights = {
                "simplicity": 50,
                "generalization": 50,
                "average_trace_fitness": 100,
                "precision": 50,
            }
        return sum(scores[key] * weights[key] for key in scores.keys())

    def fitness(self):
        scores = {
            "simplicity": self.simplicity(),
            "generalization": self.generalization(),
            "average_trace_fitness": self.perc_fit_traces(),
            "precision": self.precision(),
        }
        return self.weighted_score(scores)


if __name__ == "__main__":
    process_tree = ProcessTree(
        Operator.SEQUENCE,
        children=[
            ProcessTree(label="A"),
            ProcessTree(operator=Operator.PARALLEL, children=[
                ProcessTree(label="B"),
                ProcessTree(label="C"),
            ]),
            ProcessTree(label="D"),
        ]
    )
    
    event_log = EventLog.from_trace_list(["ACBD", "ABCD"])
    objective = SimpleWeightedScore(process_tree, event_log)
    print(objective.fitness())
    
