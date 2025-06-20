import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.FileLoader import FileLoader
from src.Discovery import Discovery
from src.utils import load_hyperparameters_from_csv


DATASET_DIR = "./logs"
OUTPUT_DIR = "./data/figure_6/"
BEST_PARAMS = "./best_parameters.csv"
MAX_GENERATIONS = 300
PERCENTAGE_OF_LOG = 0.05
OBJECTIVE = {
    "simplicity": 10,
    "refined_simplicity": 10,
    "ftr_precision": 30,
    "ftr_fitness": 50,
}

def generate_data(method: callable):
    datasets = os.listdir(DATASET_DIR)

    for dataset in datasets:
        print(f'Processing {dataset}')
        eventlog = FileLoader.load_eventlog(f"{DATASET_DIR}/{dataset}")
        method(eventlog)
        
if __name__ == "__main__":
    hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)

    # Define model
    genetic_miner = lambda log: Discovery.genetic_algorithm(
        log,
        method_name="GTM",
        export_decomposed_objective_function_path=OUTPUT_DIR,
        percentage_of_log=PERCENTAGE_OF_LOG,
        max_generations=MAX_GENERATIONS,
        **hyper_parameters,
    )
    
    generate_data(genetic_miner)