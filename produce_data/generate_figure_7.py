# add the parent directory to the system path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import os
from src.FileLoader import FileLoader
from src.Discovery import Discovery
from src.EventLog import EventLog 
from src.Evaluator import SingleEvaluator
from src.Objective import Objective
import plotly.graph_objects as go
import plotly.express as px
from src.utils import load_hyperparameters_from_csv

INPUT_DIR = "./logs/"
DATASET = "2013-cp.xes"
OUTPUT_DIR = "./data/figure_7/"

TIME_LIMT = 5 * 60
STAGNATION_LIMIT = 50
BEST_PARAMS = "./best_parameters.csv"
NUM_SAMPLES = 5
OBJECTIVE_WEIGHTS = {
    "simplicity": 10,
    "refined_simplicity": 10,
    "ftr_fitness": 50,
    "ftr_precision": 30
}


def sample_hyperparameters(hyper_parameters, num_samples):
    
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
        if isinstance(value, (int)):
            random = np.random.default_rng()
            samples = random.uniform(value - limits[key], value + limits[key], num_samples)
            sampled_hyperparameters[key] = [int(sample) for sample in samples]
        elif isinstance(value, float):
            random = np.random.default_rng()
            samples = random.uniform(value - limits[key], value + limits[key], num_samples)
            sampled_hyperparameters[key] = [round(sample, 2) for sample in samples]
    return sampled_hyperparameters

def produce_data():
    datasets = os.listdir(INPUT_DIR)
    loader = FileLoader()
    eventlogs = []
    
    best_hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)
    sampled_hyper_parameters = sample_hyperparameters(best_hyper_parameters, NUM_SAMPLES)
    data = []

    for dataset in datasets:
        if dataset == "2013-i.xes":
            continue
        loaded_log = loader.load_eventlog(f"{INPUT_DIR}/{dataset}")
        eventlogs.append(loaded_log)
    
    for eventlog in eventlogs:
        for i in range(NUM_SAMPLES):
            cur_hyper_parameters = {key: value[i] for key, value in sampled_hyper_parameters.items()}
            discovered_net = Discovery.genetic_algorithm(
                eventlog,
                mutator=best_hyper_parameters['mutator'],
                generator=best_hyper_parameters['generator'],
                objective=Objective(OBJECTIVE_WEIGHTS),
                time_limit=TIME_LIMT,
                **cur_hyper_parameters,
                stagnation_limit=STAGNATION_LIMIT,
                percentage_of_log=0.05,
            )
            
            evaluator = SingleEvaluator(
                discovered_net,
                eventlog
            )
            
            # Get the evaluation metrics
            cur_hyper_parameters['objective_fitness'] = evaluator.get_objective_fitness(OBJECTIVE_WEIGHTS) / 100
            cur_hyper_parameters['dataset'] = eventlog.name
            data.append(cur_hyper_parameters)
        
    df = pd.DataFrame(data)
    # check if the output directory exists, if not create it
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_DIR + "data.csv", index=False)
    
if __name__ == "__main__":
    produce_data()
