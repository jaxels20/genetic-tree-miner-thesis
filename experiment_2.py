import pandas
import os
from src.Discovery import Discovery
from src.EventLog import EventLog 
from src.Mutator import Mutator, TournamentMutator
from src.RandomTreeGenerator import BottomUpBinaryTreeGenerator, InjectionTreeGenerator, SequentialTreeGenerator
from src.Evaluator import SingleEvaluator
from src.Objective import Objective
import matplotlib.pyplot as plt
import seaborn as sns

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
            
    # convert the hyper parameters to a normalize 
    hyper_parameters = convert_json_to_hyperparamters(hyper_parameters)
    
    hyper_parameters['objective'] = Objective({
        "simplicity": 10,
        "generalization": 10,
        "ftr_fitness": 50,
        "ftr_precision": 30
    })
    
    DATASET_DIR = "./real_life_datasets/"
    dataset_dirs = os.listdir(DATASET_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{DATASET_DIR}{x}")]
    overall_df = pandas.DataFrame()
    for dataset_dir in dataset_dirs:
        if dataset_dir != "2013-op" and dataset_dir != "2013-cp":
            continue
        
        # Load the event log
        eventlog = EventLog.load_xes(f"{DATASET_DIR}{dataset_dir}/{dataset_dir}.xes")

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
            metrics['dataset'] = dataset_dir
            metrics['objective_fitness'] = metrics['objective_fitness'] / 100
            data.append(metrics)
        
        # Convert the data to a pandas DataFrame
        cur__df = pandas.DataFrame(data)
        # concatenate the curr ent DataFrame with the overall DataFrame
        overall_df = pandas.concat([overall_df, cur__df], ignore_index=True)
    
     # Rename the columns
    overall_df.rename(columns={
        'log_fitness': 'Replay Fitness',
        'objective_fitness': 'Objective Fitness',
        'dataset': 'Dataset',
        'precision': 'Precision',
        'simplicity': 'Simplicity',
        'generalization': 'Generalization',
    }, inplace=True)
        
    # Melt the DataFrame for Seaborn
    df_melted = overall_df.melt(id_vars='Dataset', 
                        value_vars=['Replay Fitness', 'Objective Fitness', 'Precision', 'Simplicity', 'Generalization'],
                        var_name='Metric', 
                        value_name='Score')
    
    # Use a black and white style
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 6))

    # Plot
    ax = sns.violinplot(
        data=df_melted,
        x='Dataset',
        y='Score',
        hue='Metric',
        palette='gray',   # B/W color palette
        cut=0
    )

    # Final tweaks
    #ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(title='Metric')

    # Save the plot
    plt.savefig('./experiment_2/variation_over_multiple_runs.pdf', dpi=300)

    
    
    