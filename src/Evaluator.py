import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os
import tempfile
from time import time
import pickle
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from src.EventLog import EventLog
from src.PetriNet import PetriNet
from src.Objective import Objective
from src.ProcessTree import ProcessTree

import pm4py.write as pm4py_write
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import apply as replay_fitness
from pm4py.algo.evaluation.precision.variants.etconformance_token import apply as precision
from pm4py.algo.evaluation.generalization.variants.token_based import apply as generalization
from pm4py.algo.evaluation.simplicity.variants.arc_degree import apply as simplicity
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ProcessPoolExecutor
from pm4py.convert import convert_to_process_tree as convert_to_pt


# This class can evaluate a discovered process model against an event log (only one!)
class SingleEvaluator:
    def __init__(self, pn: PetriNet, eventlog: EventLog, pt: ProcessTree):
        self.eventlog = eventlog
        self.pn = pn
        self.pt = pt

        # convert the eventlog to pm4py format
        self.pm4py_pn, self.init_marking, self.final_marking = self.pn.to_pm4py()
        self.event_log_pm4py = self.eventlog.to_pm4py()
    
    def get_evaluation_metrics(self, objective_metric_weights: dict[str, float]):
        data = {
            "simplicity": self.get_simplicity(),
            "generalization": self.get_generalization(),
            **self.get_replay_fitness(),
            "precision": self.get_precision(),
            "objective_fitness": self.get_objective_fitness(objective_metric_weights),
        }
        data["f1_score"] = self.get_f1_score(data["precision"], data["log_fitness"])
        return data    
    
    def get_simplicity(self):
        simplicity_value = simplicity(self.pm4py_pn)
        return simplicity_value
    
    def get_refined_simplicity(self):
        max_places = 100
        simplicity = len(self.pm4py_pn.places) / max_places
        simplicity = 1 - simplicity
        simplicity = max(0, simplicity)
        return simplicity 
    
    def get_generalization(self):
        generalization_value = generalization(self.event_log_pm4py, self.pm4py_pn, self.init_marking, self.final_marking)
        return generalization_value
    
    def get_replay_fitness(self):
        fitness = replay_fitness(self.event_log_pm4py, self.pm4py_pn, self.init_marking, self.final_marking)
        return fitness
    
    def get_precision(self):
        precision_value = precision(self.event_log_pm4py, self.pm4py_pn, self.init_marking, self.final_marking)
        return precision_value
    
    def get_objective_fitness(self, objective_metric_weights: dict):
        objective = Objective(objective_metric_weights)
        objective.set_event_log(self.eventlog)
        return objective.fitness(self.pt)

    
    def get_f1_score(self, precision=None, fitness=None):
        if precision is None:
            precision = self.get_precision()
        if fitness is None:
            fitness = self.get_replay_fitness()
        try:
            f1_score = 2 * (precision * fitness) / (precision + fitness)
        except ZeroDivisionError:
            f1_score = 0.0
        return f1_score

    def get_ftr_fitness(self, objective_metric_weights: dict[str, float]={"simplicity": 10, "refined_simplicity": 10, "ftr_fitness": 50, "ftr_precision": 30}):
        objective = Objective(objective_metric_weights)
        objective.set_event_log(self.eventlog)
        return objective.ftr_fitness(self.pn.to_fast_token_based_replay())
    
    def get_ftr_precision(self, objective_metric_weights: dict[str, float]={"simplicity": 10, "refined_simplicity": 10, "ftr_fitness": 50, "ftr_precision": 30}):
        objective = Objective(objective_metric_weights)
        objective.set_event_log(self.eventlog)
        return objective.ftr_precision(self.pn.to_fast_token_based_replay())

# This function discovers a process model from an event log 
# and evaluates it against the event log (calculates the metrics)
class MultiEvaluator:
    def __init__(self, event_logs: list[EventLog], methods_dict: dict, cpu_count: int = 1):
        """
        Initialize with dictionaries of Petri nets and event logs.
        Args:
        - event_logs (dict): A dictionary where keys are event log names and values are EventLog objects.
        - methods (list): A list of strings representing the discovery methods to use.
        """
        self.event_logs = event_logs # list of EventLog objects
        self.petri_nets = {method: {} for method in methods_dict.keys()} # dictionary of Petri nets with keys as discovery methods 
        self.times = {method: {} for method in methods_dict.keys()} # dictionary of times with keys as discovery methods and values a dict of event log names and PetriNet objects
        self.cpu_count = cpu_count
    
        for method, miner in methods_dict.items():
            for event_log in self.event_logs:
                print("Running discovery for", method, "on", event_log.name)
                start = time()
                pn_result = miner(event_log)
                discovery_time = time() - start
                self.petri_nets[method][event_log.name] = pn_result
                self.times[method][event_log.name] = discovery_time

        
    def evaluate_all(self, objective_metric_weights: dict[str, float] = None):
        """
        Evaluate all Petri nets against their corresponding event logs and return a DataFrame with metrics.
        """
        tasks = []

        for miner, dataset_pn_pairs in self.petri_nets.items():
            for dataset, pn in dataset_pn_pairs.items():
                # Find corresponding event log
                event_log = next(el for el in self.event_logs if el.name == dataset)
                discovery_time = self.times[miner][dataset]
                tasks.append((miner, dataset, pn, event_log, discovery_time, objective_metric_weights))

        # Use multiprocessing to evaluate in parallel
        with Pool(processes=self.cpu_count) as pool:
            results = pool.map(MultiEvaluator._evaluate_single, tasks)

        return pd.DataFrame(results)

    @staticmethod
    def _evaluate_single(args):
        miner, dataset, pn, event_log, discovery_time, objective_metric_weights = args
        evaluator = SingleEvaluator(pn, event_log)
        res = {k: round(v, 3) for k, v in evaluator.get_evaluation_metrics(objective_metric_weights).items()}
        res["dataset"] = dataset
        res["miner"] = miner
        res["time"] = discovery_time
        res["ftr_fitness"] = evaluator.get_ftr_fitness(objective_metric_weights)
        return res

    def export_petri_nets(self, output_dir, format="png"):
        """
        Export the Petri nets to the output directory.
        """
        for method, datasets in self.petri_nets.items():
            for dataset, pn in datasets.items():
                if format == "png":
                    pn.visualize(f"{output_dir}/{dataset}/{method}", format="png")
                elif format == "pdf":
                    pn.visualize(f"{output_dir}/{dataset}/{method}", format="pdf")
                else:
                    raise ValueError(f"Invalid format: {format}. Must be 'png' or 'pdf'.")

    @staticmethod
    def save_df_to_pdf(df, pdf_path):
        """
        Save the DataFrame to a single PDF figure with all datasets grouped.
        """
        table_data = []
        column_headers = ["Dataset", "Method", "F1-score", "Replay Fitness", "Precision", "Generalization", "Simplicity", "Objective fitness", "Time"]
        grouped = df.groupby('dataset')
        
        # Assing colors per dataset
        dataset_colors = ['#d9d9d9', '#ffffff']  # Light grey and white
        color_map = {}  # To store the color for each dataset
        current_color_index = 0

        for dataset, group in grouped:
            color_map[dataset] = dataset_colors[current_color_index]
            current_color_index = 1 - current_color_index  # Alternate colors

            for i, (_, row) in enumerate(group.iterrows()):
                table_data.append([
                    dataset if i == 0 else "",  # Show dataset name only in first row
                    row['miner'],
                    f"{row['f1_score']:.3f}",
                    f"{row['log_fitness']:.3f}",
                    f"{row['precision']:.3f}",
                    f"{row['generalization']:.3f}",
                    f"{row['simplicity']:.3f}",
                    f"{float(row['objective_fitness']):.3f}" if row['objective_fitness'] != "-" else "-",
                    f"{float(row['time']):.3f}" if row['time'] != "-" else "-",
                ])

        # Create the figure and add the table
        fig, ax = plt.subplots(figsize=(20, len(table_data) * 0.4 + 1))  # dynamic height
        ax.axis('off')

        # Add the table to the figure
        table = ax.table(
            cellText=table_data,
            colLabels=column_headers,
            loc='center',
            cellLoc='center',
            colLoc='center',
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col=list(range(len(column_headers))))
        
        # Make header bold
        for i, col in enumerate(column_headers):
            cell = table[0, i]
            cell.set_text_props(weight='bold')
        
        # Adjust row heights
        row_heights = 0.02  # Adjust this value to control row height
        for i in range(len(table_data) + 1):  # +1 for header row
            for j in range(len(column_headers)):
                cell = table[i, j]
                cell.set_height(row_heights)
        
        # Color the rows by dataset
        row_index = 1  # Start after header row
        for dataset, group in grouped:
            color = color_map[dataset]
            for _ in range(len(group)):
                for col in range(len(column_headers)):
                    table[row_index, col].set_facecolor(color)
                row_index += 1

        # Save the figure to the PDF
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
            plt.close(fig)

        print(f"PDF saved as {pdf_path}")
        
    @staticmethod
    def plot_monitor_data(input_dir="./monitor_analysis/data/", output_dir="./monitor_analysis/plots/"):
        # Load pickle data
        pckl_files = []
        subfolders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
        for subfolder in subfolders:
            subfolder_path = os.path.join(input_dir, subfolder)
            for file in os.listdir(subfolder_path):
                if file.endswith(".pkl"):
                    pckl_files.append(os.path.join(subfolder_path, file))
        
        # Unpack all data and prepare lines
        lines = defaultdict(dict)
        for pckl_file in pckl_files:
            with open(pckl_file, "rb") as f:
                eventlog_name, method_name, data = pickle.load(f)
            lines[eventlog_name][method_name] = data

        # Plot all data
        for eventlog_name, data in lines.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            for method_name, data in data.items():
                x = list(data.keys())
                y = list(data.values())
                ax.plot(x, y, label=f"{eventlog_name} - {method_name}", linewidth=2)
            
            # Custom plot for each event log
            ax.set_title("Fitness Evalution", fontsize=16)
            ax.set_xlabel("Generation", fontsize=12)
            ax.set_ylabel("Fitness", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(title="Method", fontsize=10)
            plt.tight_layout()
            
            # Save it
            os.makedirs(os.path.join(output_dir, eventlog_name), exist_ok=True)
            fig.savefig(os.path.join(output_dir, eventlog_name, "monitor.png"))