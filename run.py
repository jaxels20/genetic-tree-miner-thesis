from src.GeneticAlgorithm import GeneticAlgorithm
from src.Mutator import Mutator
from src.EventLog import EventLog
import time
from src.Evaluator import SingleEvaluator
from src.PetriNet import PetriNet
from src.Filtering import Filtering
import csv
import pandas as pd
from src.EventLog import EventLog

if __name__ == "__main__":
    el = EventLog.load_xes("real_life_datasets/BPI_Challenge_2013_closed_problems/BPI_Challenge_2013_closed_problems.xes")
    print(f"Unique activities: {el.unique_activities()}")
    print(f"Number of traces: {len(el.traces)}")
    
    filtered_log = Filtering.filter_eventlog_by_top_percentage_unique(el, 0.1, True)
    print(f"Unique activities after filtering: {filtered_log.unique_activities()}")
    print(f"Number of traces after filtering: {len(filtered_log.traces)}")
    
            
    