from src.GeneticAlgorithm import GeneticAlgorithm
from src.Mutator import Mutator
from src.EventLog import EventLog
from src.FileLoader import FileLoader
from src.Discovery import Discovery
from src.EventLog import EventLog
from src.Evaluator import SingleEvaluator, MultiEvaluator
from src.PetriNet import PetriNet
from src.Filtering import Filtering
import pm4py
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("./experiment_1/results_all.csv")
    MultiEvaluator.save_df_to_pdf(df, "./experiment_1/results_all.pdf")
    