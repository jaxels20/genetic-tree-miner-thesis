from src.EventLog import EventLog
from src.Objective import Objective
from src.Discovery import Discovery
from src.utils import load_hyperparameters_from_csv, calculate_percentage_of_log
import time as t
from src.Evaluator import SingleEvaluator
import json
import os

def main():
    datasets = os.listdir("./logs/")
    
    for dataset in datasets:
        el = EventLog.load_xes(f"./logs/{dataset}")
        num_unique_traces = len(el.get_unique_traces())
        percentage_of_log = calculate_percentage_of_log(num_unique_traces)
        
        print(f"Processing dataset: {dataset} with {num_unique_traces} unique traces, percentage of log: {percentage_of_log:.2f * 100}")
    
if __name__ == "__main__":
    main()