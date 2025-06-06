# add the parent directory to the system path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import random
import os
from src.FileLoader import FileLoader
from src.Discovery import Discovery
from src.Evaluator import SingleEvaluator
from src.utils import load_hyperparameters_from_csv

DATASET_DIR = "./logs/"
OUTPUT_DIR = "./data/figure_7/"

TIME_LIMT = 5*60
STAGNATION_LIMIT = 50
BEST_PARAMS = "./best_parameters.csv"
NUM_SAMPLES = 10
OBJECTIVE_WEIGHTS = {
    "simplicity": 10,
    "refined_simplicity": 10,
    "ftr_fitness": 50,
    "ftr_precision": 30
}


def sample_hyperparameters(hyper_parameters, num_samples):
    def sample_hyperparameter(lower_bound, upper_bound):
        return random.uniform(lower_bound, upper_bound)
    
    limits = {
        'random_creation_rate': 0.1,
        'elite_rate': 0.1,
        'population_size': 5,
        'tournament_size': 0.05,
        'log_filtering': 0.01,
        'tournament_rate': 0.1,
        'tournament_mutation_rate': 0.1,
    }
    sampled_hyperparameters = {}
    for key, value in hyper_parameters.items():
        if isinstance(value, int):
            lower_bound = max(value - limits[key], 0)
            upper_bound = value + limits[key]
            sampled_hyperparameters[key] = [int(sample_hyperparameter(lower_bound, upper_bound)) for i in range(num_samples)]
        elif isinstance(value, float):
            lower_bound = max(value - limits[key], 0)
            upper_bound = value + limits[key]
            sampled_hyperparameters[key] = [float(sample_hyperparameter(lower_bound, upper_bound)) for i in range(num_samples)]
        else:
            sampled_hyperparameters[key] = [value] * num_samples
    return sampled_hyperparameters

def produce_data():
    best_hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)
    sampled_hyper_parameters = sample_hyperparameters(best_hyper_parameters, NUM_SAMPLES)
    data = []
    
    datasets = os.listdir(DATASET_DIR)
    for dataset in datasets:
        print(f'Processing {dataset}')
        eventlog = FileLoader.load_eventlog(f"{DATASET_DIR}/{dataset}")
    
        for i in range(NUM_SAMPLES):
            hyperparameters_instance = {key: value[i] for key, value in sampled_hyper_parameters.items()}
            discovered_net, discovered_pt = Discovery.genetic_algorithm(
                eventlog,
                time_limit=TIME_LIMT,
                stagnation_limit=STAGNATION_LIMIT,
                **hyperparameters_instance
            )
            
            evaluator = SingleEvaluator(
                pn=discovered_net,
                eventlog=eventlog,
                pt=discovered_pt,
            )
            
            for key in ['mutator', 'objective', 'generator']:
                del hyperparameters_instance[key]
            hyperparameters_instance['objective_fitness'] = evaluator.get_objective_fitness(OBJECTIVE_WEIGHTS) / 100
            hyperparameters_instance['dataset'] = eventlog.name
            data.append(hyperparameters_instance)
        
    df = pd.DataFrame(data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_DIR + "data.csv", index=False)
    
if __name__ == "__main__":
    produce_data()
