import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from src.BatchFileLoader import BatchFileLoader
from src.Discovery import Discovery
from src.Evaluator import SingleEvaluator
from src.Filtering import Filtering


def plot_results(results, save_path, filtering_percentages):
    """ Generates and saves a line plot for filtering percentage vs. fitness difference. """
    plt.figure(figsize=(8, 5))
    plt.xlabel('Filtering Percentage')
    plt.ylabel('Fitness')
    plt.xticks(filtering_percentages)
    plt.grid(True)

    for eventlog_name, data in results.items():
        plt.plot(data.keys(), data.values(), marker='o', linestyle='-', label=eventlog_name)

    plt.legend()
    plt.savefig(save_path + '.png')
    plt.close()

def export_results(save_path, results):
    """Exports results dictionary to a CSV file with event log names as rows."""
    df = pd.DataFrame.from_dict(results, orient='index')  # Event log names as row indices
    df.index.name = "Event Log"
    df.to_csv(save_path + '.csv')

def main():
    INPUT_DIR = './real_life_datasets/'
    OUTPUT_DIR = './filtering_analysis/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filtering_percentages = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    eventlogs = [f for f in os.listdir(INPUT_DIR) if f.endswith('open_problems.xes')]

    results_top_n_filtering = defaultdict(dict)
    results_random_filtering = defaultdict(dict)
    
    for e in eventlogs:
        eventlog_name = e.split('.')[0]
        print(f"Processing event log: {eventlog_name}")
        eventlog_path = os.path.join(INPUT_DIR, e)
        _, eventlog = BatchFileLoader._load_eventlog(eventlog_path)

        # Discover and evaluate process model using full log
        # discover_log = Filtering.filter_eventlog_by_top_percentage_unique(eventlog, 0.01)  # Pre-run filtering for performance
        pn = Discovery.run_discovery("Inductive Miner", eventlog)  # Pre-run discovery for performance
        evaluator = SingleEvaluator(pn, eventlog)
        full_log_fitness = evaluator.get_replay_fitness()['log_fitness']
        
        results_top_n_filtering[eventlog_name] = {}
        results_random_filtering[eventlog_name] = {}
        for p in filtering_percentages:
            # Top N filtering
            filtered_eventlog = Filtering.filter_eventlog_by_top_percentage_unique(eventlog, p)
            pn = Discovery.run_discovery(
                "Genetic Miner", 
                filtered_eventlog,
                random_creation_rate=0.2,
                crossover_rate=0.2,
                mutation_rate=0.3,
                elite_rate=0.3,
                min_fitness=None,
                max_generations=200,
                stagnation_limit=200,
                time_limit=None,
                population_size=100)
            evaluator = SingleEvaluator(pn, eventlog)
            fitness = evaluator.get_replay_fitness()['log_fitness']
            results_top_n_filtering[eventlog_name][p] = round(full_log_fitness - fitness, 4)

    # Export and visualize results
    export_results(os.path.join(OUTPUT_DIR, 'top_traces_filtering'), results_top_n_filtering)
    export_results(os.path.join(OUTPUT_DIR, 'random_filtering'), results_random_filtering)

    plot_results(results_top_n_filtering, os.path.join(OUTPUT_DIR, 'top_traces_filtering'), filtering_percentages)
    plot_results(results_random_filtering, os.path.join(OUTPUT_DIR, 'random_filtering'), filtering_percentages)


if __name__ == "__main__":
    main()
