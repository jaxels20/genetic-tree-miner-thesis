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
    df = pd.read_csv("./experiment_1/miner_results/results_split.csv")
    df = df[['dataset','f1_score','log_fitness','precision','objective_fitness','generalization','simplicity', 'time']]
    df.to_csv("./experiment_1/miner_results/results_split.csv", index=False)
    # df.to_latex('./experiment_1/BottomUpBinaryGenerator_results.tex', index=False, float_format="%.2f")