import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.Discovery import Discovery
from src.FileLoader import FileLoader
from src.Objective import Objective
from src.Evaluator import SingleEvaluator
from src.utils import load_hyperparameters_from_csv
import pandas as pd


LOGS = "./logs/"
OUTPUT_PATH = "./data/figure_10"
TIME_LIMIT = 60*5
STAGNATION_LIMIT = 50

def calc_ftr_precision(evaluator: SingleEvaluator) -> float:
    return evaluator.get_ftr_precision()

def calc_ftr_fitness(evaluator: SingleEvaluator) -> float:
    return evaluator.get_ftr_fitness()

def calc_simplicity(evaluator: SingleEvaluator) -> float:
    return evaluator.get_simplicity()

def calc_refined_simplicity(evaluator: SingleEvaluator) -> float:
    return evaluator.get_refined_simplicity()

if __name__ == "__main__":
    datasets = os.listdir(LOGS)
    results = []
    
    for dataset in datasets:
        event_log = FileLoader.load_eventlog(f"{LOGS}/{dataset}")
        hyperparameters = load_hyperparameters_from_csv("./best_parameters.csv")
        del hyperparameters["objective"]
        
        # weight_shares = [0.25, 0.4, 0.55, 0.7, 0.85]
        weight_shares = [1.0]
        metrics = ["ftr_precision", "ftr_fitness", "simplicity", "refined_simplicity"]
        metric_functions = {
            "ftr_precision": calc_ftr_precision,
            "ftr_fitness": calc_ftr_fitness,
            "simplicity": calc_simplicity,
            "refined_simplicity": calc_refined_simplicity,
        }

        for m in metrics:
            for i, w in enumerate(weight_shares):
                objective_weights = {m: w}
                remaining_weight = (1 - w) / len(metrics)
                other_metrics = [x for x in metrics if x != m]
                for om in other_metrics:
                    objective_weights[om] = remaining_weight
                
                objective = Objective(objective_weights)                    
                net, tree = Discovery.genetic_algorithm(
                    event_log=event_log,
                    time_limit=TIME_LIMIT,
                    stagnation_limit=STAGNATION_LIMIT,
                    objective=objective,
                    **hyperparameters,
                )
                
                evaluator = SingleEvaluator(net, event_log, tree)
                results.append({
                    "metric": m,
                    "dataset": event_log.name,
                    "weight_share": w,
                    "value": metric_functions[m](evaluator),
                })
            
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_PATH}/data.csv", index=False)
