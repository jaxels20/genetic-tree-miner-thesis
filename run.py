from src.GeneticAlgorithm import GeneticAlgorithm
from src.Mutator import Mutator
from src.EventLog import EventLog
import time
from src.Evaluator import SingleEvaluator, MultiEvaluator
from src.PetriNet import PetriNet
from src.Filtering import Filtering
import csv
import pandas as pd
from src.EventLog import EventLog

if __name__ == "__main__":
    df = pd.read_csv("./experiment_1/results.csv")
    MultiEvaluator.save_df_to_pdf(df, "./experiment_1/results.pdf")
    
    