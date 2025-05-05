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
    df = pd.read_csv("./experiment_1/final_results.csv")
    datasets_to_keep = ['Nasa', '2017', '2019', '2020-pl', '2013-i', '2013-op', '2020-ptc', '2020-id', '2020-rfp']
    df = df[df['dataset'].isin(datasets_to_keep)]
    
    MultiEvaluator.save_df_to_pdf(df, "./experiment_1/results.pdf")
    
    