from src.FileLoader import FileLoader
from src.Evaluator import MultiEvaluator
from src.Mutator import Mutator, TournamentMutator
from src.Objective import Objective
from src.RandomTreeGenerator import BottomUpBinaryTreeGenerator, SequentialTreeGenerator, InjectionTreeGenerator
from src.Discovery import Discovery
import os
import pickle
import plotly.graph_objects as go
from copy import deepcopy
from experiment_1 import load_hyperparameters_from_csv
import plotly.express as px
from itertools import cycle
from pprint import pprint
import pandas as pd

INPUT_DIR = "./real_life_datasets/"
OUTPUT_DIR = "./experiment_5" 
NUM_RUNS = 1
BEST_PARAMS = "./best_parameters.csv"
TIME_LIMIT = None
STAGNATION_LIMIT = None
MAX_GENERATIONS = 200
colors = cycle(px.colors.qualitative.Pastel2)
color_map = {
    "Sequential": next(colors),
    "BottomUp": next(colors),
    "Injection": next(colors),
    "Tourn.": next(colors),
    "NonTourn.": next(colors)
}
marker_map = {
    "Sequential": "circle",
    "BottomUp": "square",
    "Injection": "triangle-up",
    "Tourn.": "triangle-down",
    "NonTourn.": "star"
}


def generate_monitors():
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()
    eventlogs = []
    hyperparams = load_hyperparameters_from_csv(BEST_PARAMS)


    for dataset_dir in dataset_dirs:
        # Assume only one file per directory
        if dataset_dir not in ["2013-cp", "2013-op", "2013-i", "Sepsis", "RTF", "2012"]:
            continue
          
        xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
        if len(xes_file) == 0:
            continue
        elif len(xes_file) == 1:
            loaded_log = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")
            eventlogs.append(loaded_log)
        else:
            raise ValueError("More than one xes file in the directory")

    methods_dict = {}

    # # Change the mutator to TournamentMutator if you want to use tournament selection
    curr_hyperparams = deepcopy(hyperparams)
    curr_hyperparams["mutator"] = TournamentMutator(
        random_creation_rate=curr_hyperparams['random_creation_rate'], 
        crossover_rate=curr_hyperparams['crossover_rate'], 
        mutation_rate=curr_hyperparams['mutation_rate'], 
        elite_rate=curr_hyperparams['elite_rate'], 
        tournament_size=curr_hyperparams['tournament_size'])
    
    curr_hyperparams["method_name"] = "Tourn."
    
    methods_dict["Tourn."] = lambda log: Discovery.genetic_algorithm(
            log,
            export_monitor_path=f"{OUTPUT_DIR}",
            time_limit=TIME_LIMIT,
            stagnation_limit=STAGNATION_LIMIT,
            max_generations=MAX_GENERATIONS,
            **curr_hyperparams
        )

    # NonTournament
    curr_hyperparams1 = deepcopy(hyperparams)
    curr_hyperparams1["mutator"] = Mutator(
        random_creation_rate=curr_hyperparams1['random_creation_rate'], 
        crossover_rate=curr_hyperparams1['crossover_rate'], 
        mutation_rate=curr_hyperparams1['mutation_rate'], 
        elite_rate=curr_hyperparams1['elite_rate'])
    
    curr_hyperparams1["method_name"] = "NonTourn."
    methods_dict["NonTourn."] = lambda log: Discovery.genetic_algorithm(
        log,
        export_monitor_path=f"{OUTPUT_DIR}",
        time_limit=TIME_LIMIT,
        stagnation_limit=STAGNATION_LIMIT,
        max_generations=MAX_GENERATIONS,
        **curr_hyperparams1
    )
    
    # Sequential
    curr_hyperparams2 = deepcopy(hyperparams)
    curr_hyperparams2["generator"] = SequentialTreeGenerator()
    curr_hyperparams2["method_name"] = "Sequential"
    methods_dict["Sequential"] = lambda log: Discovery.genetic_algorithm(
        log,
        export_monitor_path=f"{OUTPUT_DIR}",
        time_limit=TIME_LIMIT,
        stagnation_limit=STAGNATION_LIMIT,
        max_generations=MAX_GENERATIONS,
        **curr_hyperparams2
    )
    # BottomUp
    curr_hyperparams3 = deepcopy(hyperparams)
    curr_hyperparams3["generator"] = BottomUpBinaryTreeGenerator()
    curr_hyperparams3["method_name"] = "BottomUp"
    methods_dict["BottomUp"] = lambda log: Discovery.genetic_algorithm(
        log,
        export_monitor_path=f"{OUTPUT_DIR}",
        time_limit=TIME_LIMIT,
        stagnation_limit=STAGNATION_LIMIT,
        max_generations=MAX_GENERATIONS,
        **curr_hyperparams3
    )
    # Injection
    curr_hyperparams4 = deepcopy(hyperparams)
    curr_hyperparams4["generator"] = InjectionTreeGenerator(curr_hyperparams4["log_filtering"])
    curr_hyperparams4["method_name"] = "Injection"
    methods_dict["Injection"] = lambda log: Discovery.genetic_algorithm(
        log,
        export_monitor_path=f"{OUTPUT_DIR}",
        time_limit=TIME_LIMIT,
        stagnation_limit=STAGNATION_LIMIT,
        max_generations=MAX_GENERATIONS,
        **curr_hyperparams4
    )
    

    for i in range(NUM_RUNS):
        multi_evaluator = MultiEvaluator(eventlogs, methods_dict)

    #multi_evaluator.plot_monitor_data()


def visualize(folder_path, mutator: bool):
    df = pd.DataFrame()
    pkl_files = [f for f in os.listdir(folder_path + "/monitors") if f.endswith('.pkl')]
    fig = go.Figure()
    for pkl_file in pkl_files:
        if not "Tourn" in pkl_file and mutator:
            continue
        if "Tourn" in pkl_file and not mutator:
            continue
        
        with open(os.path.join(folder_path, "monitors", pkl_file), 'rb') as f:
            dataset_name, method_name, result_dict = pickle.load(f)
            
        generations = list(result_dict.keys())
        fitness_values = list(result_dict.values())
        
        # remove every second element from generations and fitness_values
        generations = generations[::2]
        fitness_values = fitness_values[::2]
        
        temp_df = pd.DataFrame({
            "Generation": generations,
            "Fitness": fitness_values,
            "Method": method_name
        })
        df = pd.concat([df, temp_df], ignore_index=True)
    # group by method and generation and mean fitness 
    df = df.groupby(["Method", "Generation"]).mean().reset_index()
    
    fig = go.Figure()
    for method in df["Method"].unique():
        method_df = df[df["Method"] == method]
        fig.add_trace(go.Scatter
            (
                x=method_df["Generation"],
                y=method_df["Fitness"],
                mode='lines+markers',
                name=method,
                line=dict(color=color_map[method], width=1),
                marker=dict(symbol=marker_map[method], size=8, line=dict(width=1, color='black'))
            )
        )
    
    fig.update_layout(title=None,
                      xaxis_title="Generation",
                      yaxis_title="Objective Fitness",
                      width=900,
                      template='simple_white',
                      height=600,
                      legend=dict(
                          font=dict(size=14),
                          orientation="v",
                          yanchor="bottom",
                          y=0.01,
                          xanchor="right",
                          x=0.99
                      )
    )
    
    # set the x axis to be 0 to max generations
    fig.update_xaxes(range=[0, 100])
    if mutator:
        fig.write_image(f"{folder_path}/mutator_comparison.png")
    else:
        fig.write_image(f"{folder_path}/generator_comparison.png")
            
def visualize_all():
    folder = "./experiment_5"
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    
    for subfolder in subfolders:
        visualize(subfolder, mutator=True)
        visualize(subfolder, mutator=False)
        

if __name__ == "__main__":    
    generate_monitors()
    visualize_all()