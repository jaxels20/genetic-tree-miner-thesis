import os
import csv
from src.FileLoader import FileLoader
from src.Objective import Objective
from src.PetriNet import PetriNet
from src.ProcessTree import ProcessTree
from pm4py.convert import convert_to_process_tree as convert_to_pt

INPUT_DIR = "./real_life_datasets/"
INPUT_PN_DIR = "./splitminer/pnml_models/"
OUTPUT_DIR = "./splitminer/"
OUTPUT_CSV = "./experiment_1/results_temp_inductive.csv"

if __name__ == "__main__":
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()

    rows = []
    for dataset_dir in dataset_dirs:
        print(f"Processing dataset: {dataset_dir}")
        xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
        if len(xes_file) == 0:
            continue
        elif len(xes_file) == 1:
            our_log = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")
            our_pn = PetriNet.from_pnml(f"{INPUT_PN_DIR}{dataset_dir}.pnml")
            pm4py_pn, im, fm = our_pn.to_pm4py()

            objective = Objective({"simplicity": 10, "refined_simplicity": 10, "ftr_fitness": 50, "ftr_precision": 30})
            objective.set_event_log(our_log)

            simplicity = objective.simplicity(pm4py_pn)
            generalization = objective.generalization(pm4py_pn, im, fm)
            perc_fit_traces = objective.perc_fit_traces(pm4py_pn, im, fm)
            average_trace_fitness = objective.average_trace_fitness(pm4py_pn, im, fm)
            log_fitness = objective.log_fitness(pm4py_pn, im, fm)
            percentage_of_fitting_traces = objective.percentage_of_fitting(pm4py_pn, im, fm)
            precision = objective.precision(pm4py_pn, im, fm)
            f1_score = 2 * (precision * log_fitness) / (precision + log_fitness + 1e-09)
            objective_fitness = objective.fitness_from_pn(pm4py_pn, im, fm)
            
            if dataset_dir == "2017":
                for trace in our_log.traces:
                    for event in trace.events:
                        name = event.activity
                        name = name.strip()
                        event.activity = name              
            
            row = [
                round(simplicity, 3),
                round(generalization, 3),
                round(perc_fit_traces, 3),
                round(average_trace_fitness, 3),
                round(log_fitness, 3),
                round(percentage_of_fitting_traces, 3),
                round(precision, 3),
                round(objective_fitness, 3),
                round(f1_score, 3),
                dataset_dir,
                "Split Miner",  # or any miner name you used
                "-"
            ]
            rows.append(row)
        else:
            raise ValueError("More than one xes file in the directory")
    
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "simplicity", "generalization", "perc_fit_traces", "average_trace_fitness", "log_fitness",
            "percentage_of_fitting_traces", "precision", "objective_fitness", "f1_score", "dataset", "miner", "time"
        ])
        writer.writerows(rows)

    print(f"Saved results to {OUTPUT_CSV}")