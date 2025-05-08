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
# TIME_LIMIT = 60
CPU_COUNT = 1
OBJECTIVE_WEIGHTS = {"simplicity": 10, "refined_simplicity": 10, "ftr_fitness": 50, "ftr_precision": 30}

def convert_json_to_hyperparamters(hyper_parameters: dict):
    # total = hyper_parameters['random_creation_rate'] + hyper_parameters['mutation_rate'] + hyper_parameters['crossover_rate'] + hyper_parameters['elite_rate']
    # hyper_parameters['random_creation_rate'] = hyper_parameters['random_creation_rate'] / total
    # hyper_parameters['mutation_rate'] = hyper_parameters['mutation_rate'] / total
    # hyper_parameters['crossover_rate'] = hyper_parameters['crossover_rate'] / total
    # hyper_parameters['elite_rate'] = hyper_parameters['elite_rate'] / total
    
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
        hyper_parameters['random_creation_rate'],
        hyper_parameters['elite_rate'],
        hyper_parameters['tournament_size'],
        hyper_parameters['tournament_rate'],
        hyper_parameters['tournament_mutation_rate']
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
            # hyper_parameters['percentage_of_log'] = float(row['percentage_of_log'])
            hyper_parameters['generator'] = row['generator']
            # hyper_parameters['mutator'] = row['mutator']
            
            hyper_parameters['tournament_size'] = float(row['tournament_size'])
            hyper_parameters['tournament_rate'] = float(row['tournament_rate'])
            hyper_parameters['tournament_mutation_rate'] = float(row['tournament_mutation_rate'])
            
            if hyper_parameters['generator'] == 'InductiveNoiseInjectionGenerator':
                hyper_parameters['log_filtering'] = float(row['log_filtering'])
            
            hyper_parameters['method_name'] = "Genetic Miner"
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
        "Genetic Miner (1 minute)": lambda log: Discovery.genetic_algorithm(
            log,
            stagnation_limit=STAGNATION_LIMIT,
            time_limit=60,
            percentage_of_log=0.05,
            **hyperparams,
        ),
        "Genetic Miner (5 minutes)": lambda log: Discovery.genetic_algorithm(
            log,
            stagnation_limit=STAGNATION_LIMIT,
            time_limit=60*5,
            percentage_of_log=0.05,
            **hyperparams,
        ),
        "Genetic Miner (30 minutes)": lambda log: Discovery.genetic_algorithm(
            log,
            stagnation_limit=STAGNATION_LIMIT,
            time_limit=60*30,
            percentage_of_log=0.05,
            **hyperparams,
        ),
    }
    
    # Run the methods on each event log
    multi_evaluator = MultiEvaluator(eventlogs, methods_dict, CPU_COUNT)
    results_df = multi_evaluator.evaluate_all(OBJECTIVE_WEIGHTS)
    
    return results_df
    
def consolidate_results(input_dir):
    # list all csv files in a directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    df = pd.DataFrame()
    for file in csv_files:
        loaded_df = pd.read_csv(input_dir + file)
        df = pd.concat([df, loaded_df], ignore_index=True)
    
    return df
    
if __name__ == "__main__":
    # Run some experiments producing csv files
    produce_csv = True
    if produce_csv:
        result_df = run_experiment()
        result_df.to_csv(OUTPUT_DIR + "/csvs/" + "results_genetic_all.csv", index=False)
    
    consolidate_results_bool = False
    if consolidate_results_bool:
        df = consolidate_results("./experiment_1/csvs/")
        column_order = ['dataset', 'miner', 'f1_score', 'log_fitness', 'precision', 'objective_fitness', 'generalization', 'simplicity', 'time']
        df = df[column_order]
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
        df.sort_values(by=['Dataset', 'Discovery Method'], inplace=True)
        
        # Output results
        df.to_latex(OUTPUT_DIR + "results_all_test.tex", index=False, escape=False, float_format="%.2f")
        df.to_csv(OUTPUT_DIR + "results_all.csv", index=False)