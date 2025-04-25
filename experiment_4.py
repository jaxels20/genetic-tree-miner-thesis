import pandas as pd
import numpy as np
import os
from itertools import cycle
from src.Discovery import Discovery
from src.EventLog import EventLog 
from src.Evaluator import SingleEvaluator
import plotly.graph_objects as go
import plotly.express as px
from experiment_1 import load_hyperparameters_from_csv

TIME_LIMT = 60*5
STAGNATION_LIMIT = 50
MAX_GENERATIONS = 3
BEST_PARAMS = "./best_parameters.csv"
DATASET = "./real_life_datasets/2013-cp/2013-cp.xes"
NUM_DATA_POINTS = 2
OBJECTIVE = {
    "simplicity": 10,
    "refined_simplicity": 10,
    "ftr_fitness": 50,
    "ftr_precision": 30
}


def sample_hyperparameters(hyper_parameters, num_samples):
    sampled_hyperparameters = {}
    for key, value in hyper_parameters.items():
        if isinstance(value, (int, float)):
            # Sample from a normal distribution with mean = value and std = 0.1 * value
            random = np.random.default_rng()
            sampled_hyperparameters[key] = random.normal(value, 0.1 * value, num_samples)
        else:
            sampled_hyperparameters[key] = [value] * num_samples
    return sampled_hyperparameters


def get_data():
    best_hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)
    sampled_hyper_parameters = sample_hyperparameters(best_hyper_parameters, NUM_DATA_POINTS)
    eventlog = EventLog.load_xes(DATASET)
    data = []
    
    for i in range(int(NUM_DATA_POINTS)):
        cur_hyper_parameters = {key: value[i] for key, value in sampled_hyper_parameters.items()}
        discovered_net = Discovery.genetic_algorithm(
            eventlog,
            max_generations=MAX_GENERATIONS,
            **cur_hyper_parameters
        )
        
        evaluator = SingleEvaluator(
            discovered_net,
            eventlog
        )
        
        # Get the evaluation metrics
        metrics = evaluator.get_evaluation_metrics(OBJECTIVE)
        metrics['objective_fitness'] = metrics['objective_fitness'] / 100
        data.append(metrics)
        
    df = pd.DataFrame(data)
    df.to_csv("./experiment_4/experiment_4_results.csv", index=False)
    return df
    
if __name__ == "__main__":
    data = get_data()
    print(data)