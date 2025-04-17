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


INPUT_DIR = "./real_life_datasets/"
OUTPUT_DIR = "./experiment_5" 
NUM_RUNS = 5
BEST_PARAMS = "./best_parameters.csv"
TIME_LIMIT = 5*60
STAGNATION_LIMIT = 15
colors = cycle(px.colors.qualitative.Pastel2)
color_map = {
    "Tourn.": next(colors),
    "NonTourn.": next(colors),
    "Sequential": next(colors),
    "BottomUp": next(colors),
    "Injection": next(colors)
}
marker_map = {
    "Tourn.": "circle",
    "NonTourn.": "square",
    "Sequential": "diamond",
    "BottomUp": "cross",
    "Injection": "x"
}


def generate_monitors():
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()
    eventlogs = []
    hyperparams = load_hyperparameters_from_csv(BEST_PARAMS)


    for dataset_dir in dataset_dirs:
        # Assume only one file per directory
        if dataset_dir != "2013-op":
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
    # curr_hyperparams = deepcopy(hyperparams)
    # curr_hyperparams["mutator"] = TournamentMutator(
    #     random_creation_rate=curr_hyperparams['random_creation_rate'], 
    #     crossover_rate=curr_hyperparams['crossover_rate'], 
    #     mutation_rate=curr_hyperparams['mutation_rate'], 
    #     elite_rate=curr_hyperparams['elite_rate'], 
    #     tournament_size=curr_hyperparams['tournament_size'])
    
    # curr_hyperparams["method_name"] = "Tourn."
    
    # methods_dict["Tourn."] = lambda log: Discovery.genetic_algorithm(
    #         log,
    #         export_monitor_path=f"{OUTPUT_DIR}",
    #         time_limit=TIME_LIMIT,
    #         stagnation_limit=STAGNATION_LIMIT,
    #         **curr_hyperparams
    #     ),

    # # NonTournament
    # curr_hyperparams1 = deepcopy(hyperparams)
    # curr_hyperparams1["mutator"] = Mutator(
    #     random_creation_rate=curr_hyperparams1['random_creation_rate'], 
    #     crossover_rate=curr_hyperparams1['crossover_rate'], 
    #     mutation_rate=curr_hyperparams1['mutation_rate'], 
    #     elite_rate=curr_hyperparams1['elite_rate'])
    # curr_hyperparams1["method_name"] = "NonTourn."
    # methods_dict["NonTourn."] = lambda log: Discovery.genetic_algorithm(
    #     log,
    #     export_monitor_path=f"{OUTPUT_DIR}",
    #     time_limit=TIME_LIMIT,
    #     stagnation_limit=STAGNATION_LIMIT,
    #     **curr_hyperparams1
    # )
    
    # Sequential
    curr_hyperparams2 = deepcopy(hyperparams)
    curr_hyperparams2["generator"] = SequentialTreeGenerator()
    curr_hyperparams2["method_name"] = "Sequential"
    methods_dict["Sequential"] = lambda log: Discovery.genetic_algorithm(
        log,
        export_monitor_path=f"{OUTPUT_DIR}",
        time_limit=TIME_LIMIT,
        stagnation_limit=STAGNATION_LIMIT,
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
        **curr_hyperparams4
    )
    

    for i in range(NUM_RUNS):
        multi_evaluator = MultiEvaluator(eventlogs, methods_dict)

    #multi_evaluator.plot_monitor_data()


def visualize(folder_path):
    seen_methods = set()
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    fig = go.Figure()
    for pkl_file in pkl_files:
        with open(os.path.join(folder_path, pkl_file), 'rb') as f:
            dataset_name, method_name, result_dict = pickle.load(f)
            
        generations = list(result_dict.keys())
        fitness_values = list(result_dict.values())
        fig.add_trace(
            go.Scatter(
                x=generations, 
                y=fitness_values, 
                mode='lines+markers', 
                marker=dict(size=5, color=color_map[method_name], symbol=marker_map[method_name]),
                name=f"{method_name}",
                showlegend=method_name not in seen_methods,
))
        seen_methods.add(method_name)

    fig.update_layout(title=None,
                      xaxis_title="Generation",
                      yaxis_title="Objective Fitness",
                      legend_title="Method",
                      width=900,
                      template='simple_white',
                      height=600,)
    
    
    fig.write_image(f"{OUTPUT_DIR}/fitness_over_generations.pdf")
            
            
            
if __name__ == "__main__":    
    generate_monitors()
    
    path = "./experiment_5/monitors/2013-op/"
    visualize(path)