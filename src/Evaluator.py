import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os
import tempfile
from time import time
import pickle
from collections import defaultdict

from src.EventLog import EventLog
from src.Discovery import Discovery
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
    def __init__(self, pn: PetriNet, eventlog: EventLog):
        self.eventlog = eventlog
        self.pn = pn

        # convert the eventlog to pm4py format
        self.pm4py_pn, self.init_marking, self.final_marking = self.pn.to_pm4py()
        self.event_log_pm4py = self.eventlog.to_pm4py()
    
    def get_evaluation_metrics(self):
        data = {
            "simplicity": self.get_simplicity(),
            "generalization": self.get_generalization(),
            **self.get_replay_fitness(),
            "precision": self.get_precision(),
            "objective_fitness": self.get_objective_fitness(),
            #"exact_matching_precision": self.get_exact_matching(type="precision"),
        }
        data["f1_score"] = self.get_f1_score(data["precision"], data["log_fitness"])
        return data    
    
    def get_simplicity(self):
        simplicity_value = simplicity(self.pm4py_pn)
        return simplicity_value
    
    def get_generalization(self):
        generalization_value = generalization(self.event_log_pm4py, self.pm4py_pn, self.init_marking, self.final_marking)
        return generalization_value
    
    def get_replay_fitness(self):
        fitness = replay_fitness(self.event_log_pm4py, self.pm4py_pn, self.init_marking, self.final_marking)
        return fitness
    
    def get_precision(self):
        precision_value = precision(self.event_log_pm4py, self.pm4py_pn, self.init_marking, self.final_marking)
        return precision_value
    
    def get_objective_fitness(self):
        pm4py_pt = convert_to_pt(self.pm4py_pn, self.init_marking, self.final_marking )
        our_pt = ProcessTree.from_pm4py(pm4py_pt)
        calculator = Objective(self.eventlog)
        return calculator.fitness(our_pt)
    
    def get_exact_matching(self, type):
        """Runs the jbpt library to calculate the entropy-based precision metric.
        
        Args:
        - type (str): The type of metric to calculate. Must be either "precision" or "recall".
        """
        
        jar_path = os.path.join("src", "EntropyBasedMetrics", "jbpt-pm", "entropia", "jbpt-pm-entropia-1.7.jar")
        
        # Use a safe temporary directory
        temp_dir = tempfile.gettempdir()
        temp_path_pn = os.path.join(temp_dir, "petrinet.pnml")
        temp_path_el = os.path.join(temp_dir, "eventlog.xes")

        # rename all tau transitions to empty string so that they are recognized as silent transitions
        for t in self.pm4py_pn.transitions:
            if t.label == None:
                t.label = ""
        
        # Convert event log and Petri net to required formats
        self.eventlog.to_xes(temp_path_el)
        pm4py_write.write_pnml(self.pm4py_pn, self.init_marking, self.final_marking, temp_path_pn)

        # Java command to run jBPT Entropy-based Precision
        if type == "precision":
            type_flag = "-emp"
        elif type == "recall":
            type_flag = "-emr"
        else:
            raise ValueError("Invalid type. Must be 'precision' or 'recall'.")
        
        command = [
            "java", "-jar", jar_path,
            type_flag,
            "-s",
            "-rel=" + temp_path_el,
            "-ret=" + temp_path_pn
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("Error running the jbpt library:", e.stderr.strip())
            
        # Remove the temporary files
        os.remove(temp_path_pn)
        os.remove(temp_path_el)
        
        return float(result.stdout)
    
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

# Define a helper function that will handle evaluation for a single Petri net and event log pair
def evaluate_single(miner: str, dataset: str, petri_net: PetriNet, event_log: EventLog):
    evaluator = SingleEvaluator(petri_net, event_log)
    
    # Get metrics and round to 4 decimal places
    metrics = {k: round(v, 3) for k, v in evaluator.get_evaluation_metrics().items()}
    metrics['miner'] = miner
    metrics['dataset'] = dataset
    
    return metrics

# This function discovers a process model from an event log 
# and evaluates it against the event log (calculates the metrics)
class MultiEvaluator:
    def __init__(self, event_logs: list[EventLog], methods_dict: dict):
        """
        Initialize with dictionaries of Petri nets and event logs.
        Args:
        - event_logs (dict): A dictionary where keys are event log names and values are EventLog objects.
        - methods (list): A list of strings representing the discovery methods to use. can be "alpha", "heuristic", "inductive", "GNN"
        """
        self.event_logs = event_logs # list of event logs with keys as event log names and values as EventLog objects
        self.petri_nets = {method: {} for method in methods_dict.keys()} # dictionary of Petri nets with keys as discovery methods 
        self.times = {method: {} for method in methods_dict.keys()} # dictionary of times with keys as discovery methods
        # and values a dict of event log names and PetriNet objects
        for method, miner in methods_dict.items():
            for event_log in self.event_logs:
                print("Running discovery for", method, "on", event_log.name)
                start = time()
                pn_result = miner(event_log)
                discovery_time = time() - start
                self.petri_nets[method][event_log.name] = pn_result
                self.times[method][event_log.name] = discovery_time
        
    def evaluate_all(self):
        """
        Evaluate all Petri nets against their corresponding event logs using multiprocessing,
        and return a DataFrame with metrics.
        """
        results = []
        # Iterate through each miner type and dataset in petri_nets
        for miner, datasets in self.petri_nets.items():
            for dataset, pn in datasets.items():
                for i in range(len(self.event_logs)):
                    event_log = self.event_logs[i]
                    res = evaluate_single(miner, dataset, pn, event_log)
                    results.append(res)
        
        for dataset in results:
            dataset_name = dataset['dataset']
            dataset["time"] = self.times[dataset["miner"]][dataset_name]
        
        return pd.DataFrame(results)

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

    def save_df_to_pdf(self, df, pdf_path):
        """
        Save the DataFrame to a single PDF figure with all datasets grouped.
        """
        table_data = []
        column_headers = ["Dataset", "Method", "Replay Fitness", "Precision", "Generalization", "Simplicity", "Objective fitness", "Time"]
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
                    f"{row['log_fitness']:.3f}",
                    f"{row['precision']:.3f}",
                    f"{row['generalization']:.3f}",
                    f"{row['simplicity']:.3f}",
                    f"{row['objective_fitness']:.3f}",
                    # f"{row['exact_matching_precision']:.3f}",
                    f"{row['time']:.3f}",
                ])

        # Create the figure and add the table
        fig, ax = plt.subplots(figsize=(12, 6))
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
        row_heights = 0.06  # Adjust this value to control row height
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