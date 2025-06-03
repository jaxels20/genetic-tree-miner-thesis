from src.EventLog import EventLog
from src.Objective import Objective
from src.Discovery import Discovery
from src.utils import load_hyperparameters_from_csv, calculate_percentage_of_log
import time as t
from src.Evaluator import SingleEvaluator
from src.Filtering import Filtering
import json
import os

def main():
    # datasets = os.listdir("./logs/")
    
    # for dataset in datasets:

    #     el = EventLog.load_xes(f"./logs/{dataset}")
    #     num_unique_traces = el.get_num_unique_traces()
    #     print(f"Event log {dataset} has {num_unique_traces} unique traces")
        
    #     filtered_el = Filtering.filter_eventlog_by_top_percentage_unique(el, 0.05, include_all_activities=True)
    #     filtered_el.get_num_unique_traces()
    #     print(f"Filtered event log {dataset} has {filtered_el.get_num_unique_traces()} unique traces")
    
    traces = ["ABC", "ABC", "ABC", "ABC", "AB", "AB", "AB", "A", "A", "B", "B"]
    manuel_el = EventLog.from_trace_list(traces)
    filtered_manuel_el = Filtering.filter_eventlog_by_top_percentage_unique(manuel_el, 0.5, include_all_activities=True)
    print(f"Manual event log has {filtered_manuel_el} unique traces")
    
    
if __name__ == "__main__":
    main()