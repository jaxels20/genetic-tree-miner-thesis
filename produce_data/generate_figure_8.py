import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import os
from src.FileLoader import FileLoader
from src.Discovery import Discovery
from src.Evaluator import SingleEvaluator
from src.utils import load_hyperparameters_from_csv

INPUT_DIR = "./logs/"
OUTPUT_DIR = "./data/figure_8/"

MAX_GENERATIONS = 100
STAGNATION_LIMIT = None
BEST_PARAMS = "./best_parameters.csv"
OBJECTIVE_WEIGHTS = {
    "simplicity": 10,
    "refined_simplicity": 10,
    "ftr_fitness": 50,
    "ftr_precision": 30
}
percentage_of_logs = [0.01, 0.05, 0.1, 0.3, 0.5]

def produce_data():
    datasets = os.listdir(INPUT_DIR)
    loader = FileLoader()
    eventlogs = []
    
    best_hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)
    data = []

    for dataset in datasets:
        loaded_log = loader.load_eventlog(f"{INPUT_DIR}/{dataset}")
        eventlogs.append(loaded_log)
    
    for eventlog in eventlogs:
        for percentage_of_log in percentage_of_logs:
            discovered_net = Discovery.genetic_algorithm(
                eventlog,
                max_generations=MAX_GENERATIONS,
                stagnation_limit=STAGNATION_LIMIT,
                percentage_of_log=percentage_of_log,
                **best_hyper_parameters,
            )
            
            evaluator = SingleEvaluator(
                discovered_net,
                eventlog
            )
            
            curr_data = {}
            # Get the evaluation metrics
            curr_data['objective_fitness'] = evaluator.get_objective_fitness(OBJECTIVE_WEIGHTS) / 100
            curr_data['dataset'] = eventlog.name
            curr_data['percentage_of_log'] = percentage_of_log
            data.append(curr_data)
        
    df = pd.DataFrame(data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_DIR + "data.csv", index=False)
    
if __name__ == "__main__":
    produce_data()
