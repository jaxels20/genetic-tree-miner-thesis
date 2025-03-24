from src.BatchFileLoader import BatchFileLoader
from src.Evaluator import MultiEvaluator
import os

INPUT_DIR = "./controlled_scenarios/"
OUTPUT_DIR = "./controlled_scenarios_results/" 
METHODS = ["Genetic Miner"]
NUM_WORKERS = 1

if __name__ == "__main__":
    dataset_dirs = os.listdir(INPUT_DIR)
    # remove all files from the list
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    eventlogs = {} # name: eventlog
    loader = BatchFileLoader(cpu_count=NUM_WORKERS)
    
    for dataset_dir in dataset_dirs:
        temp_eventlogs = loader.load_all_eventlogs(f"{INPUT_DIR}{dataset_dir}")
        
        # check that there is only one event log in the directory
        assert len(temp_eventlogs) == 1, f"Expected one event log in {dataset_dir}, got {len(temp_eventlogs)}"
        eventlogs[dataset_dir] = next(iter(temp_eventlogs.values()))
    
    #Create and evaluate the MultiEvaluator
    multi_evaluator = MultiEvaluator(
        eventlogs,
        methods=METHODS,
        population_size=100,
        max_generations=100,
        random_creation_rate=0.2,
        crossover_rate=0.75,
        mutation_rate=0.2,
        elite_rate=0.05,
        tournament_size=0.25,
        min_fitness=None,
        stagnation_limit=None,
        time_limit=None,
    )
    
    results_df = multi_evaluator.evaluate_all(num_cores=NUM_WORKERS)
    multi_evaluator.save_df_to_pdf(results_df, OUTPUT_DIR + "results.pdf")
    multi_evaluator.export_petri_nets(OUTPUT_DIR)