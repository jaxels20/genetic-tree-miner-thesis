import csv
import os
from src.Discovery import Discovery
from src.Mutator import Mutator, TournamentMutator
from src.RandomTreeGenerator import BottomUpRandomBinaryGenerator, FootprintGuidedSequentialGenerator, InductiveNoiseInjectionGenerator, InductiveMinerGenerator
from src.Evaluator import MultiEvaluator, SingleEvaluator
from src.FileLoader import FileLoader
from src.Objective import Objective

INPUT_DIR = "./real_life_datasets/"
OUTPUT_DIR = "./experiment_1/"
SPLIT_MINER_DIR = "./experiment_1/split_miner/"
BEST_PARAMS = "./best_parameters.csv"

def convert_json_to_hyperparamters(hyper_parameters: dict):
    total = hyper_parameters['random_creation_rate'] + hyper_parameters['mutation_rate'] + hyper_parameters['crossover_rate'] + hyper_parameters['elite_rate']
    hyper_parameters['random_creation_rate'] = hyper_parameters['random_creation_rate'] / total
    hyper_parameters['mutation_rate'] = hyper_parameters['mutation_rate'] / total
    hyper_parameters['crossover_rate'] = hyper_parameters['crossover_rate'] / total
    hyper_parameters['elite_rate'] = hyper_parameters['elite_rate'] / total
    
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
    if hyper_parameters['mutator'] == 'Tournament':
        hyper_parameters['mutator'] = TournamentMutator(
            hyper_parameters['random_creation_rate'],
            hyper_parameters['elite_rate'],
            hyper_parameters['tournament_size']
        )
    elif hyper_parameters['mutator'] == 'NonTournament':
        hyper_parameters['mutator'] = Mutator(
            hyper_parameters['random_creation_rate'],
            hyper_parameters['crossover_rate'],
            hyper_parameters['mutation_rate'],
            hyper_parameters['elite_rate']
        )
    else:
        raise ValueError("Invalid mutator type")
    
    return hyper_parameters

def load_hyperparameters_from_csv(path: str):
    hyper_parameters = {}
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            hyper_parameters['random_creation_rate'] = float(row['random_creation_rate'])
            hyper_parameters['mutation_rate'] = float(row['mutation_rate'])
            hyper_parameters['crossover_rate'] = float(row['crossover_rate'])
            hyper_parameters['elite_rate'] = float(row['elite_rate'])
            hyper_parameters['population_size'] = int(row['population_size'])
            hyper_parameters['percentage_of_log'] = float(row['percentage_of_log'])
            hyper_parameters['generator'] = row['generator']
            hyper_parameters['mutator'] = row['mutator']
            
            if hyper_parameters['mutator'] == 'Tournament':
                hyper_parameters['tournament_size'] = float(row['tournament_size'])
            
            if hyper_parameters['generator'] == 'Injection':
                hyper_parameters['log_filtering'] = float(row['log_filtering'])
            
            hyper_parameters['method_name'] = "Genetic Miner"
            hyper_parameters['objective'] = Objective({
                "simplicity": 10,
                "refined_simplicity": 10,
                "ftr_fitness": 50,
                "ftr_precision": 30
            })

    # THIS IS WRONG BUT FOR NOW !!!!!
    hyper_parameters['log_filtering'] = 0.1


    return convert_json_to_hyperparamters(hyper_parameters)

if __name__ == "__main__":
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()
    eventlogs = []

    for dataset_dir in dataset_dirs:
        if dataset_dir not in ["2013-op", "2013-cp"]:  # delete later
            continue
        xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
        if len(xes_file) == 0:
            continue
        elif len(xes_file) == 1:
            loaded_log = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")
            eventlogs.append(loaded_log)
        else:
            raise ValueError("More than one xes file in the directory")
    
    # Load the hyperparameters
    hyperparams = load_hyperparameters_from_csv(BEST_PARAMS)

    # Define the methods to be used
    methods_dict = {
        "Genetic Miner": lambda log: Discovery.genetic_algorithm(
            log,
            stagnation_limit=50,
            time_limit=60*5,
            **hyperparams,
        ),
        "Inductive Miner": lambda log: Discovery.inductive_miner(log)
    }
    
    # Run the methods on each event log
    multi_evaluator = MultiEvaluator(eventlogs, methods_dict)
    results_df = multi_evaluator.evaluate_all({"simplicity": 10, "refined_simplicity": 10, "ftr_fitness": 50, "ftr_precision": 30})
    results_df.to_csv(OUTPUT_DIR + "results.csv", index=False)
    results_df[["miner","dataset", "simplicity","generalization","log_fitness","precision","objective_fitness","time"]].to_latex(OUTPUT_DIR + "results.tex", index=False, float_format="%.2f")