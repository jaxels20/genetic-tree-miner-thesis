import csv
import os
import pandas as pd
import time
from multiprocessing import cpu_count
from src.Discovery import Discovery
from src.Mutator import Mutator, TournamentMutator
from src.RandomTreeGenerator import BottomUpRandomBinaryGenerator, FootprintGuidedSequentialGenerator, InductiveNoiseInjectionGenerator, InductiveMinerGenerator
from src.EventLog import EventLog
from src.Evaluator import SingleEvaluator
from src.Objective import Objective

# Data parameters 
DATASET_DIR = "./real_life_datasets/"

# Genetic Miner Configuration
BEST_PARAMS = "./best_parameters.csv"
TIME_LIMIT = 60*5
STAGNATION_LIMIT = 50
PERCENTAGE_OF_LOG = 0.05
OBJECTIVE = {
    "simplicity": 15,
    "refined_simplicity": 15,
    "ftr_f1_score": 70,
}

# Result generation parameters
GENERATE_RESULT_DF = False
NUM_DATA_POINTS = 5
OUTPUT_DIR = "./experiment_1/"
RESULT_FILE_NAME = OUTPUT_DIR + 'genetic_miner_test_run.csv'
MEDIAN_RUN_FILE_NAME = OUTPUT_DIR + 'miner_results/' + 'results_test_run.csv' # only alter the last string

# Output petri nets
OUTPUT_PETRI_NETS = False
PETRI_NETS_SAVE_PATH = "./genetic_miner_nets/" 

# Consolidation
CONSOLIDATE_MINER_RESULTS = True
CONSOLIDATED_RESULTS_FILE_NAME = "consolidated_results"   # dont include extension

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

def generate_data(method: callable, runs: int, results_file_name: str, output_petri_nets: bool):    
    dataset_dirs = os.listdir(DATASET_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{DATASET_DIR}{x}")]

    for dataset_dir in dataset_dirs:
        if dataset_dir not in ['2013-cp', '2013-op']:
            continue
        eventlog = EventLog.load_xes(f"{DATASET_DIR}{dataset_dir}/{dataset_dir}.xes")

        data = []
        for i in range(runs):
            print(f"Running discovery on dataset: {dataset_dir} iteration: {i}")
            start = time.time()
            discovered_net = method(eventlog)
            time_taken = time.time() - start
            
            # Export the discovered net to a file
            if output_petri_nets:
                os.makedirs(PETRI_NETS_SAVE_PATH + "pdfs/", exist_ok=True)
                discovered_net.visualize(PETRI_NETS_SAVE_PATH + f"/pdfs/{dataset_dir}_{i}")
                os.makedirs(PETRI_NETS_SAVE_PATH + "/pnmls/", exist_ok=True)
                discovered_net.to_pnml(PETRI_NETS_SAVE_PATH + f"/pnmls/{dataset_dir}_{i}")
            
            evaluator = SingleEvaluator(
                discovered_net,
                eventlog
            )
            
            # Get the evaluation metrics
            fitness = evaluator.get_replay_fitness()['log_fitness']
            precision = evaluator.get_precision()
            
            metrics = {}
            metrics['Dataset'] = dataset_dir
            metrics['Discovery Method'] = "GM"
            metrics['Model'] = i
            metrics['Log Fitness'] = fitness
            metrics['Precision'] = precision
            metrics['F1 Score'] = evaluator.get_f1_score(precision, fitness)
            metrics['Objective Fitness'] = evaluator.get_objective_fitness(OBJECTIVE)
            metrics['Generalization'] = evaluator.get_generalization()
            metrics['Simplicity'] = evaluator.get_simplicity()
            metrics['Time (s)'] = time_taken
            data.append(metrics)
            
            cur_df = pd.DataFrame(data)        
            if os.path.exists(results_file_name):
                read_df = pd.read_csv(results_file_name)
                cur_df = pd.concat([read_df, cur_df], ignore_index=True)
            
            cur_df.to_csv(results_file_name, index=False)
    
def consolidate_results(input_dir):
    # list all csv files in a directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    df = pd.DataFrame()
    for file in csv_files:
        loaded_df = pd.read_csv(input_dir + file)
        df = pd.concat([df, loaded_df], ignore_index=True)
    
    # df manipulation
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
        .groupby('Dataset')['Objective Fitness']
        .apply(lambda x: (x - x.median()).abs().idxmin())    # idxmin returns the index of the first occurrence of the minimum value
    )
    df = df.loc[median_idx].reset_index(drop=True)
    
    if os.path.exists(output_csv_name):
        existing_df = pd.read_csv(output_csv_name)
        df = pd.concat([df, existing_df])
    df.to_csv(output_csv_name, index=False)
    
if __name__ == "__main__":
    if GENERATE_RESULT_DF:
        # convert the hyper parameters to a normalize
        hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)
        hyper_parameters['objective'] = Objective(OBJECTIVE)   # DELETE DELETE DELETE DELETE DELETE LATER DELEEEEEEEEEEEEETE
        
        # Define model
        genetic_miner = lambda log: Discovery.genetic_algorithm(
            log,
            time_limit=TIME_LIMIT,
            stagnation_limit=STAGNATION_LIMIT,
            percentage_of_log=PERCENTAGE_OF_LOG,
            **hyper_parameters,
        )
        # Generate df with results over runs and write to csv
        generate_data(
            method = genetic_miner,
            runs=NUM_DATA_POINTS,
            results_file_name=RESULT_FILE_NAME,
            output_petri_nets=OUTPUT_PETRI_NETS
        )
        # Take the median of multiple runs and write to csv
        get_genetic_miner_median_run(input_file=RESULT_FILE_NAME, output_csv_name=MEDIAN_RUN_FILE_NAME)
    
    if CONSOLIDATE_MINER_RESULTS:
        df = consolidate_results(OUTPUT_DIR + "miner_results/")
        df.to_csv(OUTPUT_DIR + "consolidated_results/" + CONSOLIDATED_RESULTS_FILE_NAME + ".csv", index=False, float_format="%.2f")
        df.to_latex(OUTPUT_DIR + "consolidated_results/" + CONSOLIDATED_RESULTS_FILE_NAME + ".tex", index=False, escape=False, float_format="%.2f")