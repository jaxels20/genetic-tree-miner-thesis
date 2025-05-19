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
from experiment_1 import get_genetic_miner_median_run

if __name__ == "__main__":
    get_genetic_miner_median_run("./experiment_1/random_results/results_genetic_miner_5_runs_new_objective.csv", "./median_run.csv")
    df = pd.read_csv("./median_run.csv")
    df = df[['Dataset','F1 Score','Log Fitness','Precision','Objective Fitness','Generalization','Simplicity', 'Time (s)']]
    df.to_latex('./objective_results.tex', index=False, float_format="%.2f")