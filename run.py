from src.GeneticAlgorithm import GeneticAlgorithm
from src.Mutator import Mutator
from src.EventLog import EventLog
from src.FileLoader import FileLoader
from src.Discovery import Discovery
from src.EventLog import EventLog
from src.Evaluator import SingleEvaluator, MultiEvaluator
from src.PetriNet import PetriNet
from src.Filtering import Filtering
from src.Objective import Objective
import pm4py
import pandas as pd
import os

if __name__ == "__main__":
    df = pd.read_csv("./experiment_1/archive/results_genetic_5_runs.csv")
    med = df.groupby('Dataset')['Objective Fitness'].transform('median')
    filtered_df = df[df['Objective Fitness'] == med]
    
    filtered_df["Refined Simplicity"] = (1 - filtered_df["Refined Simplicity"]) * 100
    filtered_df["Refined Simplicity"] = filtered_df["Refined Simplicity"].round(0).astype(int)
    filtered_df.to_csv("./experiment_1/csvs/results_genetic.csv", index=False)
    