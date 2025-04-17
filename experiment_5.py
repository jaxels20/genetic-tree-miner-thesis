from src.FileLoader import FileLoader
from src.Evaluator import MultiEvaluator
from src.Mutator import Mutator, TournamentMutator
from src.Objective import Objective
from src.RandomTreeGenerator import BottomUpBinaryTreeGenerator, SequentialTreeGenerator, InjectionTreeGenerator
from src.Discovery import Discovery
import os
import pickle
import plotly.graph_objects as go
INPUT_DIR = "./real_life_datasets/"
OUTPUT_DIR = "./experiment_5" 
NUM_RUNS = 5

def generate_monitors():
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()
    eventlogs = []

    for dataset_dir in dataset_dirs:
        # Assume only one file per directory
        if dataset_dir != "2013-op":
            continue
          
        xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
        if len(xes_file) == 0:
            continue
        elif len(xes_file) == 1:
            loaded_log = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")
            eventlogs.append(loaded_log)
        else:
            raise ValueError("More than one xes file in the directory")

    methods_dict = {
        "Genetic Miner (Random Initial - NonTournament)": lambda log: Discovery.genetic_algorithm(
            log,
            method_name="Genetic Miner (Random Initial - NonTournament)",
            mutator=Mutator(random_creation_rate=0.2, crossover_rate=0.3, mutation_rate=0.3, elite_rate=0.2),
            generator=BottomUpBinaryTreeGenerator(),
            objective=Objective(metric_weights={"simplicity": 20, "refined_simplicity": 20, "ftr_fitness": 100, "ftr_precision": 50}),
            percentage_of_log=0.1,
            population_size=5,
            export_monitor_path=f"{OUTPUT_DIR}",
            stagnation_limit=10
        ),
        # "Genetic Miner (Sequencial Initial - NonTournament)": lambda log: Discovery.genetic_algorithm(
        #     log,
        #     method_name="Genetic Miner (Sequential Initial - NonTournament)",
        #     mutator=Mutator(random_creation_rate=0.2, crossover_rate=0.3, mutation_rate=0.3, elite_rate=0.2),
        #     generator=SequentialTreeGenerator(),
        #     objective=Objective(metric_weights={"simplicity": 20, "refined_simplicity": 20, "ftr_fitness": 100, "ftr_precision": 50}),
        #     percentage_of_log=0.1,
        #     max_generations=100,
        #     population_size=100,
        #     stagnation_limit=10,
        # ),
        # "Genetic Miner (Injection Initial - NonTournament)": lambda log: Discovery.genetic_algorithm(
        #     log,
        #     method_name="Genetic Miner (Injection Initial - NonTournament)",
        #     mutator=Mutator(random_creation_rate=0.2, crossover_rate=0.3, mutation_rate=0.3, elite_rate=0.2),
        #     generator=InjectionTreeGenerator(log_filtering=0.05),
        #     objective=Objective(metric_weights={"simplicity": 20, "refined_simplicity": 20, "ftr_fitness": 100, "ftr_precision": 50}),
        #     percentage_of_log=0.1,
        #     max_generations=100,
        #     population_size=100,
        #     stagnation_limit=10
        # ),
        # "Inductive Miner": lambda log: Discovery.inductive_miner(log),
    }

    for i in range(NUM_RUNS):
        multi_evaluator = MultiEvaluator(eventlogs, methods_dict)

    #multi_evaluator.plot_monitor_data()


def visualize(folder_path):
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    fig = go.Figure()
    for pkl_file in pkl_files:
        with open(os.path.join(folder_path, pkl_file), 'rb') as f:
            dataset_name, method_name, result_dict = pickle.load(f)
            
        generations = list(result_dict.keys())
        fitness_values = list(result_dict.values())
        fig.add_trace(go.Scatter(x=generations, y=fitness_values, mode='lines+markers', name=f"{dataset_name} - {method_name}"))
    fig.update_layout(title="Fitness over generations",
                      xaxis_title="Generation",
                      yaxis_title="Fitness",
                      legend_title="Dataset - Method")
    fig.write_image(f"{OUTPUT_DIR}/fitness_over_generations.pdf")
            
            
            

if __name__ == "__main__":    
    #generate_monitors()
    
    path = "./experiment_5/monitors/2013-op/"
    visualize(path)