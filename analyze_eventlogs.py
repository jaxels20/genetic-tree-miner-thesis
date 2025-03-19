import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from src.BatchFileLoader import BatchFileLoader
from src.Discovery import Discovery
from src.Evaluator import SingleEvaluator
from src.Filtering import Filtering
import csv

def plot_results(results, save_path, filtering_percentages):
    """ Generates and saves a line plot for filtering percentage vs. fitness difference. """
    plt.figure(figsize=(8, 5))
    plt.xlabel('Filtering Percentage')
    plt.ylabel(r'$\Delta$ Fitness')
    plt.xticks(filtering_percentages, rotation=45, fontsize=8) 
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    for eventlog_name, data in results.items():
        plt.plot(data.keys(), data.values(), marker='o', linestyle='-', label=eventlog_name)

    plt.legend(fontsize=8, loc='upper right')
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=300)
    plt.close()

def export_results(save_path, results):
    """Exports results dictionary to a CSV file with event log names as rows."""
    df = pd.DataFrame.from_dict(results, orient='index')  # Event log names as row indices
    df.index.name = "Event Log"
    df.to_csv(save_path + '.csv')
    
def import_results(file_path):
    """Imports results from a CSV file and returns them as a dictionary."""
        # load csv file
    with open(file_path) as f:
        reader = csv.reader(f)
        data = list(reader)
        
    results = dict()
    filtering_percentages = [float(x) for x in data[0][1:]]
    for row in data[1:]:
        eventlog_name = row[0]
        results[eventlog_name] = dict()
        for i in range(1, len(row)):
            results[eventlog_name][filtering_percentages[i-1]] = float(row[i])
    
    return results

def main():
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    eventlogs = {} # name: eventlog
    loader = BatchFileLoader(cpu_count=NUM_WORKERS)
    
    for dataset_dir in dataset_dirs:
        temp_eventlogs = loader.load_all_eventlogs(f"{INPUT_DIR}{dataset_dir}")
        # check that there is only one event log in the directory
        # assert len(temp_eventlogs) == 1, f"Expected one event log in {dataset_dir}, got {len(temp_eventlogs)}"
        if dataset_dir not in ['BPI_Challenge_2013_open_problems']:
            continue
        
        for eventlog in temp_eventlogs.values():
            eventlogs[dataset_dir] = eventlog
    
    results_top_n_filtering = dict()
    for eventlog_name, cur_el in eventlogs.items():
        print("Processing event log:", eventlog_name)
        # Discover and evaluate process model using full log
        pn = Discovery.run_discovery(
            "Genetic Miner", 
            cur_el,
            random_creation_rate=0.2,
            crossover_rate=0.2,
            mutation_rate=0.3,
            elite_rate=0.3,
            min_fitness=None,
            max_generations=100,
            stagnation_limit=None,
            time_limit=None,
            population_size=100
        )
        evaluator = SingleEvaluator(pn, cur_el)
        full_log_fitness = evaluator.get_replay_fitness()['log_fitness']
        
        results_top_n_filtering[eventlog_name] = dict()
        for p in FILTERING_PERCENTAGES:
            # Top N filtering
            filtered_eventlog = Filtering.filter_eventlog_by_top_percentage_unique(cur_el, p, include_all_activities=False)
            pn = Discovery.run_discovery(
                "Genetic Miner", 
                filtered_eventlog,
                random_creation_rate=0.2,
                crossover_rate=0.2,
                mutation_rate=0.3,
                elite_rate=0.3,
                min_fitness=None,
                max_generations=100,
                stagnation_limit=None,
                time_limit=None,
                population_size=100
            )
            evaluator = SingleEvaluator(pn, cur_el)
            fitness = evaluator.get_replay_fitness()['log_fitness']
            results_top_n_filtering[eventlog_name][p] = round(full_log_fitness - fitness, 4)
        results_top_n_filtering[eventlog_name][1] = 0

    # Export and visualize results
    export_results(os.path.join(OUTPUT_DIR, 'top_traces_filtering'), results_top_n_filtering)
    plot_results(results_top_n_filtering, os.path.join(OUTPUT_DIR, 'top_traces_filtering'), FILTERING_PERCENTAGES)


if __name__ == "__main__":
    INPUT_DIR = './real_life_datasets/'
    OUTPUT_DIR = './filtering_analysis/'
    NUM_WORKERS = 1
    FILTERING_PERCENTAGES = [0.01, 0.05, 0.1, 0.2, 0.5]
    main()
    # results = import_results('./filtering_analysis/top_traces_filtering.csv')
    # plot_results(results, os.path.join(OUTPUT_DIR, 'test'), FILTERING_PERCENTAGES)
