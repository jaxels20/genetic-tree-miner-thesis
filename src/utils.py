from src.RandomTreeGenerator import BottomUpRandomBinaryGenerator, FootprintGuidedSequentialGenerator, InductiveNoiseInjectionGenerator, InductiveMinerGenerator
from src.Mutator import TournamentMutator
from src.Objective import Objective
from src.FileLoader import FileLoader
import csv


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
