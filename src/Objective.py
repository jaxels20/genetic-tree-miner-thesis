from ProcessTree import ProcessTree
from EventLog import EventLog
from ProcessTree import Operator
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import apply as replay_fitness
from pm4py.algo.evaluation.precision.variants.etconformance_token import apply as precision
from pm4py.algo.evaluation.generalization.variants.token_based import apply as generalization
from pm4py.algo.evaluation.simplicity.variants.arc_degree import apply as simplicity


class Objective:
    def __init__(self, process_tree: ProcessTree, event_log: EventLog):
        self.process_tree = process_tree
        self.pm4py_pn, self.inital_marking, self.final_marking = process_tree.to_pm4py_pn()
        self.eventlog = event_log
        self.event_log_pm4py = event_log.to_pm4py()
    
    # Some function that can calculate a score for a tree based on some criteria
    
    def simplicity(self):
        simplicity_value = simplicity(self.pm4py_pn)
        return simplicity_value
    
    def generalization(self):
        generalization_value = generalization(self.event_log_pm4py, self.pm4py_pn, self.inital_marking, self.final_marking)
        return generalization_value
    
    def replay_fitness(self):
        fitness = replay_fitness(self.event_log_pm4py, self.pm4py_pn, self.inital_marking, self.final_marking)
        return fitness['average_trace_fitness']
    
    def precision(self):
        precision_value = precision(self.event_log_pm4py, self.pm4py_pn, self.inital_marking, self.final_marking)
        return precision_value
    
    def weighted_average(self, scores: dict[str, float], weights: dict[str, float] = None) -> float:
        # Only works if the scores and weights have the same keys 
        # The Keys must be "simplicity", "generalization", "replay_fitness", "precision"
        
        if weights is None:
            weights = {
                "simplicity": 0.25,
                "generalization": 0.25,
                "replay_fitness": 0.25,
                "precision": 0.25
            }
        return sum(scores[key] * weights[key] for key in scores.keys())


if __name__ == "__main__":
    process_tree = ProcessTree(
        Operator.SEQUENCE,
        children=[
            ProcessTree(label="a"),
        ]
    )
    event_log = EventLog.from_trace_list(["a"])
    objective = Objective(process_tree, event_log)
    data = {
        "simplicity": objective.simplicity(),
        "generalization": objective.generalization(),
        "replay_fitness": objective.replay_fitness(),
        "precision": objective.precision(),
    }
