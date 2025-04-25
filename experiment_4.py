import pandas as pd
import numpy as np
import os
from src.FileLoader import FileLoader
from src.Discovery import Discovery
from src.EventLog import EventLog 
from src.Evaluator import SingleEvaluator
from src.Objective import Objective
import plotly.graph_objects as go
import plotly.express as px
from experiment_1 import load_hyperparameters_from_csv

INPUT_DIR = "./real_life_datasets/"
DATASET = "./real_life_datasets/2013-cp/2013-cp.xes"
TIME_LIMT = 3
STAGNATION_LIMIT = 50
MAX_GENERATIONS = 3
BEST_PARAMS = "./best_parameters.csv"
NUM_SAMPLES = 2
OBJECTIVE_WEIGHTS = {
    "simplicity": 10,
    "refined_simplicity": 10,
    "ftr_fitness": 50,
    "ftr_precision": 30
}


def sample_hyperparameters(hyper_parameters, num_samples):
    sampled_hyperparameters = {}
    for key, value in hyper_parameters.items():
        if isinstance(value, (int)):
            random = np.random.default_rng()
            samples = random.normal(value, 0.1 * value, num_samples)
            sampled_hyperparameters[key] = [int(sample) for sample in samples]
        elif isinstance(value, float):
            random = np.random.default_rng()
            samples = random.normal(value, 0.1 * value, num_samples)
            sampled_hyperparameters[key] = [round(sample, 2) for sample in samples]
    return sampled_hyperparameters

def create_plot(df):
    # Define the column used for coloring
    color_col = 'objective_fitness'

    # Drop it from the list of dimensions
    dimension_cols = [col for col in df.columns if col != color_col]

    # Define custom axis limits and ticks
    custom_ylim = {
        'random_creation_rate': [0, 1],
        'mutation_rate': [0, 1],
        'crossover_rate': [0, 1],
        'elite_rate': [0, 1],
        'population_size': [20, 70],
        'percentage_of_log': [0.01, 0.3],
        'tournament_size': [0.3, 0.5],
        'log_filtering': [0.001, 0.15],
    }

    y_tick_vals = {
        'random_creation_rate': [x / 100 for x in range(0, 102, 10)],
        'mutation_rate': [x / 100 for x in range(0, 102, 10)],
        'crossover_rate': [x / 100 for x in range(0, 102, 10)],
        'elite_rate': [x / 100 for x in range(0, 102, 10)],
        'population_size': list(range(20, 75, 5)),
        'percentage_of_log': [x / 100 for x in range(1, 31, 3)],
        'tournament_size': [x / 100 for x in range(30, 51, 2)],
        'log_filtering': [x / 1000 for x in range(1, 151, 20)],
    }

    # Build dimensions
    dimensions = []
    for col in dimension_cols:
        values = df[col].tolist()
        dim = dict(
            label=col.replace('_', ' ').title(),
            values=[x if not np.isnan(x) else None for x in values]
        )
        if col in custom_ylim:
            dim['range'] = custom_ylim[col]
            dim['tickvals'] = y_tick_vals[col]
        dimensions.append(dim)

    # Create parallel coordinates plot
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df[color_col],
                colorscale='Viridis',
                cmin=0.8,
                cmax=1.0, 
                colorbar=dict(title='Objective Fitness')
            ),
            dimensions=dimensions,
            labelside='bottom',
            labelangle=15
        )
    )

    # Layout settings
    fig.update_layout(
        font=dict(family='Arial', size=14),
        margin=dict(l=60, r=80, t=50, b=90),
        template='simple_white',
        height=500,
        width=900
    )

    # Save to file
    fig.write_image("./experiment_4/plot.pdf")

def get_data():
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()
    eventlogs = []
    
    best_hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)
    sampled_hyper_parameters = sample_hyperparameters(best_hyper_parameters, NUM_SAMPLES)
    eventlog = EventLog.load_xes(DATASET)
    data = []

    for dataset_dir in dataset_dirs:
        # Assume only one file per directory
        if dataset_dir not in ["2013-cp", "2013-op", "Sepsis"]:
            continue
          
        xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
        if len(xes_file) == 0:
            continue
        elif len(xes_file) == 1:
            loaded_log = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")
            eventlogs.append(loaded_log)
        else:
            raise ValueError("More than one xes file in the directory")
    
    for i in range(NUM_SAMPLES):
        cur_hyper_parameters = {key: value[i] for key, value in sampled_hyper_parameters.items()}
        discovered_net = Discovery.genetic_algorithm(
            eventlog,
            mutator=best_hyper_parameters['mutator'],
            generator=best_hyper_parameters['generator'],
            objective=Objective(OBJECTIVE_WEIGHTS),
            time_limit=TIME_LIMT,
            **cur_hyper_parameters,
        )
        
        evaluator = SingleEvaluator(
            discovered_net,
            eventlog
        )
        
        # Get the evaluation metrics
        cur_hyper_parameters['objective_fitness'] = evaluator.get_objective_fitness(OBJECTIVE_WEIGHTS) / 100
        data.append(cur_hyper_parameters)
        
    df = pd.DataFrame(data)
    df.to_csv("./experiment_4/experiment_4_results.csv", index=False)
    return df
    
if __name__ == "__main__":
    data = get_data()
    create_plot(data)