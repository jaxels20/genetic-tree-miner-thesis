import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import time
from src.Discovery import Discovery
from src.Evaluator import SingleEvaluator
from src.utils import load_hyperparameters_from_csv
from src.FileLoader import FileLoader

# Data parameters 
DATASET_DIR = "./logs/"

# Genetic Miner Configuration
BEST_PARAMS = "./best_parameters.csv"
TIME_LIMIT = 60*5
STAGNATION_LIMIT = 50
OBJECTIVE = {
    "simplicity": 10,
    "refined_simplicity": 10,
    "ftr_precision": 30,
    "ftr_fitness": 50,
}

NUM_DATA_POINTS = 5
OUTPUT_DIR = "./data/table_2"

def generate_data(method: callable, gtm_name, runs: int):    
    datasets = [f for f in os.listdir(DATASET_DIR) if f.endswith(".xes")]
    
    data = []
    for dataset in datasets:
        dataset_name = dataset.split(".")[0]
        eventlog = FileLoader.load_eventlog(f"{DATASET_DIR}{dataset}")

        for i in range(runs):
            print(f"Running discovery on dataset: {dataset_name} iteration: {i}")
            start = time.time()
            discovered_net, discovered_pt = method(eventlog)
            time_taken = time.time() - start
            
            os.makedirs(f"{OUTPUT_DIR}/models/GTM", exist_ok=True)
            discovered_net.to_pnml(f"{OUTPUT_DIR}/models/GTM/{dataset_name}_{gtm_name}_{i}")
            
            evaluator = SingleEvaluator(
                pn=discovered_net,
                eventlog=eventlog,
                pt=discovered_pt
            )
            
            # Get the evaluation metrics
            fitness = evaluator.get_replay_fitness()['log_fitness']
            precision = evaluator.get_precision()
            
            metrics = {}
            metrics['Dataset'] = dataset.split(".")[0]
            metrics['Discovery Method'] = gtm_name
            metrics['Model'] = i
            metrics['Log Fitness'] = fitness
            metrics['Precision'] = precision
            metrics['F1 Score'] = evaluator.get_f1_score(precision, fitness)
            metrics['Objective Fitness'] = evaluator.get_objective_fitness(OBJECTIVE)
            metrics['Generalization'] = evaluator.get_generalization()
            metrics['Simplicity'] = evaluator.get_simplicity()
            metrics['Time (s)'] = time_taken
            data.append(metrics)
            
    df = pd.DataFrame(data)
    df.to_csv(f"{OUTPUT_DIR}/evaluation_results/results_{gtm_name}.csv", index=False)
    
if __name__ == "__main__":
    # convert the hyper parameters to a normalize
    hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)
    
    # Define model
    genetic_miner_1 = lambda log: Discovery.genetic_algorithm(
        log,
        time_limit=10,
        stagnation_limit=STAGNATION_LIMIT,
        **hyper_parameters,
    )
    generate_data(
        method = genetic_miner_1,
        gtm_name="GTM-10",
        runs=NUM_DATA_POINTS,
    )
    
    genetic_miner_2 = lambda log: Discovery.genetic_algorithm(
        log,
        time_limit=60,
        stagnation_limit=STAGNATION_LIMIT,
        **hyper_parameters,
    )
    generate_data(
        method = genetic_miner_2,
        gtm_name="GTM-60",
        runs=NUM_DATA_POINTS,
    )
    
    genetic_miner_3 = lambda log: Discovery.genetic_algorithm(
        log,
        time_limit=300,
        stagnation_limit=STAGNATION_LIMIT,
        **hyper_parameters,
    )
    generate_data(
        method = genetic_miner_3,
        gtm_name="GTM-300",
        runs=NUM_DATA_POINTS,
    )

