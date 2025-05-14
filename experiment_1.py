import csv
import os
import pandas as pd
from multiprocessing import cpu_count
from src.Discovery import Discovery
from src.Mutator import Mutator, TournamentMutator
from src.RandomTreeGenerator import BottomUpRandomBinaryGenerator, FootprintGuidedSequentialGenerator, InductiveNoiseInjectionGenerator, InductiveMinerGenerator
from src.Evaluator import MultiEvaluator, SingleEvaluator
from src.FileLoader import FileLoader
from src.Objective import Objective

INPUT_DIR = "./real_life_datasets/"
OUTPUT_DIR = "./experiment_1/"
BEST_PARAMS = "./best_parameters.csv"

STAGNATION_LIMIT = 50
TIME_LIMIT = 60*5
PERCENTAGE_OF_LOG = 0.05
OBJECTIVE_WEIGHTS = {"simplicity": 10, "refined_simplicity": 10, "ftr_fitness": 50, "ftr_precision": 30}

def convert_json_to_hyperparamters(hyper_parameters: dict):    
    total = hyper_parameters['random_creation_rate'] + hyper_parameters['elite_rate'] + hyper_parameters["tournament_rate"]
    hyper_parameters['random_creation_rate'] = hyper_parameters['random_creation_rate'] / total
    hyper_parameters['elite_rate'] = hyper_parameters['elite_rate'] / total
    hyper_parameters['tournament_rate'] = hyper_parameters['tournament_rate'] / total
    
    # Convert the generator and mutator to objects
    if hyper_parameters['generator'] == 'BottomUpRandomBinaryGenerator':
        hyper_parameters['generator'] = BottomUpRandomBinaryGenerator()
    elif hyper_parameters['generator'] == 'FootprintGuidedSequentialGenerator':
        hyper_parameters['generator'] = FootprintGuidedSequentialGenerator()
    elif hyper_parameters['generator'] == 'InductiveNoiseInjectionGenerator':
        hyper_parameters['generator'] = InductiveNoiseInjectionGenerator(hyper_parameters['log_filtering'])
    elif hyper_parameters['generator'] == 'InductiveMinerGenerator':
        hyper_parameters['generator'] = InductiveMinerGenerator()
    else:
        raise ValueError("Invalid generator type")
    
    hyper_parameters['mutator'] = TournamentMutator(
        random_creation_rate = hyper_parameters['random_creation_rate'],
        elite_rate = hyper_parameters['elite_rate'],
        tournament_rate = hyper_parameters['tournament_rate'],
        tournament_size = hyper_parameters['tournament_size'],
        tournament_mutation_rate = hyper_parameters['tournament_mutation_rate']
    )
    
    return hyper_parameters

def load_hyperparameters_from_csv(path: str):
    hyper_parameters = {}
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            hyper_parameters['random_creation_rate'] = float(row['random_creation_rate'])
            hyper_parameters['elite_rate'] = float(row['elite_rate'])
            hyper_parameters['population_size'] = int(row['population_size'])
            hyper_parameters['generator'] = row['generator']
            
            hyper_parameters['tournament_size'] = float(row['tournament_size'])
            hyper_parameters['tournament_rate'] = float(row['tournament_rate'])
            hyper_parameters['tournament_mutation_rate'] = float(row['tournament_mutation_rate'])
            
            if hyper_parameters['generator'] == 'InductiveNoiseInjectionGenerator':
                hyper_parameters['log_filtering'] = float(row['log_filtering'])
            
            hyper_parameters['objective'] = Objective({
                "simplicity": 10,
                "refined_simplicity": 10,
                "ftr_fitness": 50,
                "ftr_precision": 30
            })

    return convert_json_to_hyperparamters(hyper_parameters)

def run_experiment():
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()
    eventlogs = []

    for dataset_dir in dataset_dirs:
        xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
        if len(xes_file) == 1:
            loaded_log = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")
            eventlogs.append(loaded_log)
        else:
            raise ValueError("More than one xes file in the directory")
    
    # Load the hyperparameters
    hyperparams = load_hyperparameters_from_csv(BEST_PARAMS)

    # Define the methods to be used
    methods_dict = {
        "GM 1": lambda log: Discovery.genetic_algorithm(
            log,
            method_name="GM 1",
            stagnation_limit=STAGNATION_LIMIT,
            time_limit=TIME_LIMIT,
            percentage_of_log=PERCENTAGE_OF_LOG,
            **hyperparams,
        ),
        "GM 2": lambda log: Discovery.genetic_algorithm(
            log,
            method_name="GM 2",
            stagnation_limit=STAGNATION_LIMIT,
            time_limit=TIME_LIMIT,
            percentage_of_log=PERCENTAGE_OF_LOG,
            **hyperparams,
        ),
        "GM 3": lambda log: Discovery.genetic_algorithm(
            log,
            method_name="GM 3",
            stagnation_limit=STAGNATION_LIMIT,
            time_limit=TIME_LIMIT,
            percentage_of_log=PERCENTAGE_OF_LOG,
            **hyperparams,
        ),
        "GM 4": lambda log: Discovery.genetic_algorithm(
            log,
            method_name="GM 4",
            stagnation_limit=STAGNATION_LIMIT,
            time_limit=TIME_LIMIT,
            percentage_of_log=PERCENTAGE_OF_LOG,
            **hyperparams,
        ),
        "GM 1": lambda log: Discovery.genetic_algorithm(
            log,
            method_name="GM 5",
            stagnation_limit=STAGNATION_LIMIT,
            time_limit=TIME_LIMIT,
            percentage_of_log=PERCENTAGE_OF_LOG,
            **hyperparams,
        )
    }
    
    # Run the methods on each event log
    multi_evaluator = MultiEvaluator(eventlogs, methods_dict)
    results_df = multi_evaluator.evaluate_all(OBJECTIVE_WEIGHTS)
    
    return results_df
    
def consolidate_results(input_dir):
    # list all csv files in a directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    df = pd.DataFrame()
    for file in csv_files:
        loaded_df = pd.read_csv(input_dir + file)
        df = pd.concat([df, loaded_df], ignore_index=True)
    
    # Manipulation
    df.rename(columns={
        'dataset': 'Dataset',
        'miner': 'Discovery Method',
        'f1_score': 'F1 Score',
        'log_fitness': 'Log Fitness',
        'precision': 'Precision',
        'objective_fitness': 'Objective Fitness',
        'generalization': 'Generalization',
        'simplicity': 'Simplicity',
        'time': 'Time (s)'
    }, inplace=True)
    column_order = ['Dataset', 'Discovery Method', 'F1 Score', 'Log Fitness', 'Precision', 'Generalization', 'Simplicity', 'Objective Fitness', 'Time (s)']
    df = df[column_order]
    df.sort_values(by=['Dataset', 'Discovery Method'], inplace=True)
    df['Time (s)'] = df['Time (s)'].replace('-', 0)
    
    agg_df = df.copy()
    agg_df = agg_df.groupby('Discovery Method').agg({
        'F1 Score': 'mean',
        'Log Fitness': 'mean',
        'Precision': 'mean',
        'Objective Fitness': 'mean',
        'Generalization': 'mean',
        'Simplicity': 'mean',
        'Time (s)': 'mean'
    }).reset_index()
    agg_df['Dataset'] = 'Aggregated'
    
    collected_df = pd.concat([df, agg_df], ignore_index=True)
    collected_df['Time (s)'] = collected_df['Time (s)'].round(2).astype(str)
    collected_df.loc[(collected_df['Discovery Method'] == 'SM') & (collected_df['Time (s)'] == '0.0'), 'Time (s)'] = '-'
    
    return collected_df
 
def get_genetic_miner_median_run(input_file, output_csv_name):
    df = pd.read_csv(input_file)
    median_idx = (
        df
        .groupby('dataset')['objective_fitness']
        .apply(lambda x: (x - x.median()).abs().idxmin())    # idxmin returns the index of the first occurrence of the minimum value
    )
    filtered_df = df.loc[median_idx].reset_index(drop=True)
    filtered_df.to_csv(output_csv_name, index=False)
    
if __name__ == "__main__":
    # A prerequise to run the script for experiment 2 that produces the csv file containing the results of the genetic miner over one or multiple runs
    if not os.path.exists(OUTPUT_DIR + "miner_results/results_genetic.csv"):
        get_genetic_miner_median_run("./experiment_1/results_genetic_5_runs.csv", OUTPUT_DIR + "miner_results/results_genetic.csv")
    df = consolidate_results("./experiment_1/miner_results/")
    df.to_csv(OUTPUT_DIR + "consolidated_results/results.csv", index=False, float_format="%.2f")
    df.to_latex(OUTPUT_DIR + "consolidated_results/results.tex", index=False, escape=False, float_format="%.2f")