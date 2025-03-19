from src.GeneticAlgorithm import GeneticAlgorithm
from src.Mutator import Mutator
from src.EventLog import EventLog
import time
from src.Evaluator import SingleEvaluator
from src.PetriNet import PetriNet
from src.Filtering import Filtering
import csv
import pandas as pd

if __name__ == "__main__":
    # load csv file
    with open('./filtering_analysis/top_traces_filtering.csv') as f:
        reader = csv.reader(f)
        data = list(reader)
        
    results = dict()
    filtering_percentages = [float(x) for x in data[0][1:]]
    for row in data[1:]:
        eventlog_name = row[0]
        results[eventlog_name] = dict()
        for i in range(1, len(row)):
            results[eventlog_name][filtering_percentages[i-1]] = float(row[i])
            
            
    