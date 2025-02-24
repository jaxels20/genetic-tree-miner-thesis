import pandas as pd
import matplotlib.pyplot as plt

from src.EventLog import EventLog
from src.Discovery import Discovery
from src.PetriNet import PetriNet

from pm4py.algo.evaluation.replay_fitness.variants.token_replay import apply as replay_fitness
from pm4py.algo.evaluation.precision.variants.etconformance_token import apply as precision
from pm4py.algo.evaluation.generalization.variants.token_based import apply as generalization
from pm4py.algo.evaluation.simplicity.variants.arc_degree import apply as simplicity
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ProcessPoolExecutor


# This class can evaluate a discovered process model against an event log (only one!)
class SingleEvaluator:
    def __init__(self, pm4py_pn, init_marking, final_marking, eventlog: EventLog):
        self.eventlog = eventlog
        
        self.pm4py_pn = pm4py_pn
        self.init_marking = init_marking
        self.final_marking = final_marking

        # convert the eventlog to pm4py format
        self.event_log_pm4py = self.eventlog.to_pm4py()
    
    def get_evaluation_metrics(self):
        data = {
            "simplicity": self.get_simplicity(),
            "generalization": self.get_generalization(),
            **self.get_replay_fitness(),
            "precision": self.get_precision(),
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
def evaluate_single(miner: str, dataset: str, petri_net, initial_marking, end_marking, event_log: EventLog):
    evaluator = SingleEvaluator(petri_net, initial_marking, end_marking, event_log)
    
    # Get metrics and round to 4 decimal places
    metrics = {k: round(v, 3) for k, v in evaluator.get_evaluation_metrics().items()}
    metrics['miner'] = miner
    metrics['dataset'] = dataset
    
    
    return metrics

# This function discovers a process model from an event log 
# and evaluates it against the event log (calculates the metrics)
class MultiEvaluator:
    def __init__(self, event_logs: dict, methods: list, **kwargs):
        """
        Initialize with dictionaries of Petri nets and event logs.
        Args:
        - event_logs (dict): A dictionary where keys are event log names and values are EventLog objects.
        
        - methods (list): A list of strings representing the discovery methods to use. can be "alpha", "heuristic", "inductive", "GNN"
        
        """
        self.event_logs = event_logs # dictionary of event logs with keys as event log names and values as EventLog objects
        self.petri_nets = {method: {} for method in methods} # dictionary of Petri nets with keys as discovery methods 
        # and values a dict of event log names and PetriNet objects
        for method in methods:
            for event_log_name, event_log in self.event_logs.items():
                pn_result, initial_marking, final_marking = Discovery.run_discovery(method, event_log, **kwargs)
                self.petri_nets[method][event_log_name] = (pn_result, initial_marking, final_marking)
        
    def evaluate_all(self, num_cores=None):
            """
            Evaluate all Petri nets against their corresponding event logs using multiprocessing,
            and return a DataFrame with metrics.
            """
            results = []
            
            # Use ProcessPoolExecutor for multiprocessing
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = []
                
                # Iterate through each miner type and dataset in petri_nets
                for miner, datasets in self.petri_nets.items():
                    for dataset, (pn, initital_marking, final_marking) in datasets.items():
                        # pn, initital_marking, final_marking = petri_net
                        if dataset in self.event_logs:
                            event_log = self.event_logs[dataset]
                            futures.append(
                                executor.submit(evaluate_single, 
                                                miner, 
                                                dataset, 
                                                pn, 
                                                initital_marking, 
                                                final_marking, 
                                                event_log)
                            )
                
                # Collect the results as they complete
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"Error evaluating Petri net: {e}")
            
            # Convert the list of dictionaries to a DataFrame
            return pd.DataFrame(results)

    def export_petri_nets(self, output_dir, format="png"):
        """
        Export the Petri nets to the output directory.
        """
        for method, datasets in self.petri_nets.items():
            for dataset, (pn, initial_marking, final_marking) in datasets.items():
                converted_pn = PetriNet.from_pm4py(pn)
                if format == "png":
                    converted_pn.visualize(f"{output_dir}/{dataset}/{method}", format="png")
                elif format == "pdf":
                    converted_pn.visualize(f"{output_dir}/{dataset}/{method}", format="pdf")
                else:
                    raise ValueError(f"Invalid format: {format}. Must be 'png' or 'pdf'.")

    def save_df_to_pdf(self, df, pdf_path):
        """
        Save the DataFrame to a single PDF figure with all datasets grouped.
        """
        table_data = []
        column_headers = ["Dataset", "Method", "F1-Score", "Fitness", "Precision", "Generalization", "Simplicity"]
        grouped = df.groupby('dataset')
        
        for dataset, group in grouped:
            # Add the dataset name in the first row
            for i, (_, row) in enumerate(group.iterrows()):
                if i == 0:
                    table_data.append([
                        dataset,  # Dataset name printed only in the first row
                        row['miner'],
                        f"{row['f1_score']:.3f}",
                        f"{row['log_fitness']:.3f}",
                        f"{row['precision']:.3f}",
                        f"{row['generalization']:.3f}",
                        f"{row['simplicity']:.3f}",
                    ])
                else:
                    table_data.append([
                        "",  # Empty dataset name for subsequent rows
                        row['miner'],
                        f"{row['f1_score']:.3f}",
                        f"{row['log_fitness']:.3f}",
                        f"{row['precision']:.3f}",
                        f"{row['generalization']:.3f}",
                        f"{row['simplicity']:.3f}",
                    ])

        # Create the figure and add the table
        fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.3))
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
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(column_headers))))

        # Save the figure to the PDF
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
            plt.close(fig)

        print(f"PDF saved as {pdf_path}")
        
    
