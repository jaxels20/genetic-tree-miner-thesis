from src.FileLoader import FileLoader
from src.Evaluator import MultiEvaluator
from src.Mutator import Mutator, TournamentMutator
from src.RandomTreeGenerator import BottomUpBinaryTreeGenerator, SequentialTreeGenerator, InjectionTreeGenerator
from src.Discovery import Discovery
import os

INPUT_DIR = "./real_life_datasets/"
OUTPUT_DIR = "./real_life_datasets_results/" 

if __name__ == "__main__":
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()
    eventlogs = []

    for dataset_dir in dataset_dirs:
        if not dataset_dir == "BPI_Challenge_2013_closed_problems":
            continue
        
        # Assume only one file per directory
        xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
        if len(xes_file) == 0:
            continue
        elif len(xes_file) == 1:
            loaded_log = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")
            eventlogs.append(loaded_log)
        else:
            raise ValueError("More than one xes file in the directory")

    methods_dict = {
        "Genetic_Miner": lambda log: Discovery.genetic_algorithm(
            log,
            method_name="Genetic_Miner_Random",
            mutator=Mutator(random_creation_rate=0.2, crossover_rate=0.3, mutation_rate=0.3, elite_rate=0.2),
            generator=BottomUpBinaryTreeGenerator(),
            percentage_of_log=0.1,
            max_generations=100,
            population_size=100,
            min_fitness=None,
            stagnation_limit=None,
            time_limit=None
        ),
        "Genetic_Miner_2": lambda log: Discovery.genetic_algorithm(
            log,
            method_name="Genetic_Miner_Sequential",
            mutator=Mutator(random_creation_rate=0.2, crossover_rate=0.3, mutation_rate=0.3, elite_rate=0.2),
            generator=SequentialTreeGenerator(),
            percentage_of_log=0.1,
            max_generations=100,
            population_size=100,
            min_fitness=None,
            stagnation_limit=None,
            time_limit=None
        ),
        "Genetic_Miner_3": lambda log: Discovery.genetic_algorithm(
            log,
            method_name="Genetic_Miner_Injection",
            mutator=Mutator(random_creation_rate=0.2, crossover_rate=0.3, mutation_rate=0.3, elite_rate=0.2),
            generator=InjectionTreeGenerator(),
            percentage_of_log=0.1,
            max_generations=100,
            population_size=100,
            min_fitness=None,
            stagnation_limit=None,
            time_limit=None
        )
    }

    multi_evaluator = MultiEvaluator(eventlogs, methods_dict)
    results_df = multi_evaluator.evaluate_all()
    multi_evaluator.save_df_to_pdf(results_df, OUTPUT_DIR + "results.pdf")
    multi_evaluator.export_petri_nets(OUTPUT_DIR)
    multi_evaluator.plot_monitor_data()