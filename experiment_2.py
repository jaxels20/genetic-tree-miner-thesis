import pandas
import os
from src.Discovery import Discovery
from src.EventLog import EventLog 
from src.Mutator import Mutator, TournamentMutator
from src.RandomTreeGenerator import BottomUpBinaryTreeGenerator, InjectionTreeGenerator, SequentialTreeGenerator
from src.Evaluator import SingleEvaluator
from src.Objective import Objective
import matplotlib.pyplot as plt

hyper_parameters = {
    'random_creation_rate': 0.019132300011790924, 
    'mutation_rate': 0.8677916440719196, 
    'crossover_rate': 0.3741045790050315, 
    'elite_rate': 0.31502212661512324, 
    'population_size': 32, 
    'percentage_of_log': 0.22147542851738136, 
    'generator': 'Injection', 
    'mutator': 'NonTournament', 
    'log_filtering': 0.026817932672967698
    }



def convert_json_to_hyperparamters(hyper_parameters: dict):
    total = hyper_parameters['random_creation_rate'] + hyper_parameters['mutation_rate'] + hyper_parameters['crossover_rate'] + hyper_parameters['elite_rate']
    hyper_parameters['random_creation_rate'] = hyper_parameters['random_creation_rate'] / total
    hyper_parameters['mutation_rate'] = hyper_parameters['mutation_rate'] / total
    hyper_parameters['crossover_rate'] = hyper_parameters['crossover_rate'] / total
    hyper_parameters['elite_rate'] = hyper_parameters['elite_rate'] / total
    
    # Convert the generator and mutator to objects
    if hyper_parameters['generator'] == 'BottomUpBinary':
        hyper_parameters['generator'] = BottomUpBinaryTreeGenerator()
    elif hyper_parameters['generator'] == 'Sequential':
        hyper_parameters['generator'] = SequentialTreeGenerator()
    elif hyper_parameters['generator'] == 'Injection':
        hyper_parameters['generator'] = InjectionTreeGenerator(hyper_parameters['log_filtering'])
    else:
        raise ValueError("Invalid generator type")
    if hyper_parameters['mutator'] == 'Tournament':
        hyper_parameters['mutator'] = TournamentMutator(
            hyper_parameters['random_creation_rate'],
            hyper_parameters['crossover_rate'],
            hyper_parameters['mutation_rate'],
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
    
    

if __name__ == "__main__":
    
    eventlog = EventLog.load_xes("real_life_datasets/BPI_Challenge_2013_open_problems/BPI_Challenge_2013_open_problems.xes")
    
    # convert the hyper parameters to a normalize 
    
    hyper_parameters = convert_json_to_hyperparamters(hyper_parameters)
    
    hyper_parameters['objective'] = Objective({
        "simplicity": 10,
        "generalization": 10,
        "ftr_fitness": 50,
        "ftr_precision": 30
    })
    data = []
    for i in range(10):
        discovered_net = Discovery.genetic_algorithm(
            eventlog,
            time_limit=60*5,
            stagnation_limit=15,
            **hyper_parameters
        )
        
        evaluator = SingleEvaluator(
            discovered_net,
            eventlog
        )
        
        # Get the evaluation metrics
        metrics = evaluator.get_evaluation_metrics({
            "simplicity": 10,
            "generalization": 10,
            "ftr_fitness": 50,
            "ftr_precision": 30
        })
        metrics['dataset'] = 'BPI_Challenge_2013_open_problems'
        metrics['objective_fitness'] = metrics['objective_fitness'] / 100
        data.append(metrics)
    
    # Convert the data to a pandas DataFrame
    df = pandas.DataFrame(data)
    
    # Prepare data for plotting
    datasets = df['dataset'].unique()
    positions = []
    plot_data = []

    spacing = 1  # space between datasets
    width = 0.3  # space between metrics in same dataset
    pos = 1

    for dataset in datasets:
        subset = df[df['dataset'] == dataset]
        plot_data.append(subset['log_fitness'].values)
        positions.append(pos)
        pos += width
        plot_data.append(subset['objective_fitness'].values)
        positions.append(pos)
        pos += spacing  # move to next dataset group

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(plot_data, positions=positions, showmeans=True, showmedians=True)

    # Set x-ticks in the middle of each dataset group
    dataset_positions = [positions[i] + width / 2 for i in range(0, len(positions), 2)]
    ax.set_xticks(dataset_positions)
    ax.set_xticklabels(datasets, rotation=45, ha='right')

    # Y-axis between 0 and 1
    #ax.set_ylim(0, 1)

    # Add legend manually
    ax.legend([parts['bodies'][0], parts['bodies'][1]], ['log_fitness', 'objective_fitness'])

    plt.title("Violin Plot of Metrics by Dataset")
    plt.tight_layout()
    plt.show()

    
    
    