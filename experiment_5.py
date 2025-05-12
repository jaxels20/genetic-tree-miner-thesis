from src.FileLoader import FileLoader
from src.Evaluator import MultiEvaluator
from src.Mutator import Mutator, TournamentMutator
from src.Objective import Objective
from src.RandomTreeGenerator import BottomUpRandomBinaryGenerator, FootprintGuidedSequentialGenerator, InductiveNoiseInjectionGenerator, InductiveMinerGenerator
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
MAX_GENERATIONS = 300

def generate_monitors():
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()
    eventlogs = []
    hyperparams = load_hyperparameters_from_csv(BEST_PARAMS)


    for dataset_dir in dataset_dirs:        
        xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
        if len(xes_file) == 1:
            loaded_log = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")
            eventlogs.append(loaded_log)
        else:
            raise ValueError("None or more than one xes file in the directory")

    methods_dict = {}
    hyperparams["method_name"] = "Genetic Miner"
    methods_dict["Genetic Miner"] = lambda log: Discovery.genetic_algorithm(
            log,
            export_monitor_path=f"{OUTPUT_DIR}",
            time_limit=TIME_LIMIT,
            stagnation_limit=STAGNATION_LIMIT,
            max_generations=MAX_GENERATIONS,
            percentage_of_log=0.05,
            **hyperparams,
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
                # line=dict(color=color_map[method], width=1),
                # marker=dict(symbol=marker_map[method], size=8, line=dict(width=1, color='black'))
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

def visualize_paper_figure():
    subfolders = [f.path for f in os.scandir(OUTPUT_DIR) if f.is_dir()]
    data = []
    
    for subfolder in subfolders:
        pkl_files = [f for f in os.listdir(subfolder + "/monitors") if f.endswith('.pkl')]
        
        for pkl_file in pkl_files:
            with open(os.path.join(subfolder, "monitors", pkl_file), 'rb') as f:
                dataset_name, method_name, result_dict = pickle.load(f)
                
            generations = list(result_dict.keys())
            fitness_values = list(result_dict.values())
            
            generations = generations[::5]
            fitness_values = fitness_values[::5]
            
            for i in range(len(generations)):
                data.append(
                    {
                        "Generation": generations[i],
                        "Fitness": fitness_values[i],
                        "Method": method_name,
                        "Dataset": dataset_name
                    }
                )
            
    # create a dataframe
    df = pd.DataFrame(data)
    
    # sort on Fitness
    df = df.sort_values(by=["Fitness"], ascending=False)
    
    # group by dataset and mean fitness
    df = df.groupby(["Dataset", "Generation"], as_index=False).mean(numeric_only=True)
    
    colorblind_colors = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
    "#117733", "#332288", "#88CCEE", "#44AA99",
    "#661100", "#6699CC"
]

    marker_symbols = [
        "circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down",
        "triangle-left", "triangle-right", "star", "hexagram", "hourglass", "bowtie", "asterisk"
    ]

    # Step 1: Compute max fitness per dataset
    dataset_order = (
        df.groupby("Dataset")["Fitness"]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # Step 2: Plot traces in sorted order
    fig = go.Figure()
    for i, dataset in enumerate(dataset_order):
        dataset_df = df[df["Dataset"] == dataset]
        fig.add_trace(go.Scatter(
            x=dataset_df["Generation"],
            y=dataset_df["Fitness"],
            mode='lines+markers',
            name=f"{dataset}",
            line=dict(color=colorblind_colors[i % len(colorblind_colors)]),
            marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=8)
        ))

    # Step 3: Update layout
    fig.update_layout(
        title=None,
        xaxis_title="Generation",
        yaxis_title="Objective Fitness",
        width=900,
        height=600,
        template='simple_white',
        legend=dict(
            font=dict(size=14),
            orientation="v",
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    # write the file to the output directory
    fig.write_image(f"{OUTPUT_DIR}/paper_figure.pdf")
    
    
if __name__ == "__main__":    
    # generate_monitors()
    visualize_paper_figure()
