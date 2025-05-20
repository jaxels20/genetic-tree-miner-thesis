from src.RandomTreeGenerator import BottomUpRandomBinaryGenerator
from src.Discovery import Discovery
from src.EventLog import EventLog
from src.FileLoader import FileLoader
import os
import pickle
import plotly.graph_objects as go
from experiment_1 import load_hyperparameters_from_csv
import pandas as pd

DATASETS_DIR = "./real_life_datasets/"
OUTPUT_DIR = "./data/figure_5c/inductive_tree_generator/"
NUM_RUNS = 1
BEST_PARAMS = "./best_parameters.csv"
PERCENTAGE_OF_LOG = 0.05
MAX_GENERATIONS = 300

def generate_monitors(method: callable):
    datasets = os.listdir(DATASETS_DIR)

    for dataset in datasets:
        print(f'Processing {dataset}')
        eventlog = FileLoader.load_eventlog(f"{DATASETS_DIR}/{dataset}")
        for i in range(NUM_RUNS):
            method(eventlog)   # Each run will export monitor object to specified export path

if __name__ == "__main__":
    hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)

    # Define model
    genetic_miner = lambda log: Discovery.genetic_algorithm(
        log,
        method_name="Genetic_miner",
        export_monitor_path=OUTPUT_DIR,
        percentage_of_log=PERCENTAGE_OF_LOG,
        max_generations=MAX_GENERATIONS,
        **hyper_parameters,
    )
    
    generate_monitors(genetic_miner)