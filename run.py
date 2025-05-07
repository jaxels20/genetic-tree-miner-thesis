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

INPUT_DIR = "./real_life_datasets/"
INPUT_PN_DIR = "./splitminer/pnml_models/"
OUTPUT_DIR = "./splitminer/"
OUTPUT_CSV = "./experiment_1/results_temp_inductive.csv"

if __name__ == "__main__":
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()

    for dataset_dir in dataset_dirs:

        xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
        if len(xes_file) == 0:
            continue
        elif len(xes_file) == 1:
            our_log = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")
            our_pn = PetriNet.from_pnml(f"{INPUT_PN_DIR}{dataset_dir}.pnml")
            pm4py_pn, im, fm = our_pn.to_pm4py()
            
            if dataset_dir == "2017":
                for trace in our_log.traces:
                    for event in trace.events:
                        name = event.activity
                        name = name.strip()
                        event.activity = name
            
            objective = Objective({"simplicity": 10, "refined_simplicity": 10, "ftr_fitness": 50, "ftr_precision": 30})
            objective.set_event_log(our_log)
            try:
                fit = objective.fitness_from_pn(pm4py_pn, im, fm)
                print(f"Fitness: {fit} on {dataset_dir}")
            except Exception as e:
                print(f"Error calculating fitness: {e}")
                continue
    
    