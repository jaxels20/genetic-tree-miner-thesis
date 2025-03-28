from src.BatchFileLoader import BatchFileLoader
from src.Evaluator import MultiEvaluator
from src.Filtering import Filtering
import os

INPUT_DIR = "./real_life_datasets/"
OUTPUT_DIR = "./real_life_datasets_results/" 
METHODS = ["Genetic Miner", "Inductive Miner"]
NUM_WORKERS = 1

if __name__ == "__main__":
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    eventlogs = {} # name: eventlog
    loader = BatchFileLoader(cpu_count=NUM_WORKERS)
    
    for dataset_dir in dataset_dirs:
        temp_eventlogs = loader.load_all_eventlogs(f"{INPUT_DIR}{dataset_dir}")
        # check that there is only one event log in the directory
        for eventlog in temp_eventlogs.values():
            eventlogs[dataset_dir] = eventlog
    
    #Create and evaluate the MultiEvaluator
    multi_evaluator = MultiEvaluator(
        eventlogs, 
        methods=METHODS,
        percentage_of_log=0.1,
        max_generations=100,
        population_size=100,
        tournament_size=0.25,
        random_creation_rate=0.2,
        crossover_rate=0.3,
        mutation_rate=0.3,
        elite_rate=0.2,
        min_fitness=None,
        stagnation_limit=None,
        time_limit=None,
    )
    
    results_df = multi_evaluator.evaluate_all()
    multi_evaluator.save_df_to_pdf(results_df, OUTPUT_DIR + "results.pdf")
    multi_evaluator.export_petri_nets(OUTPUT_DIR)