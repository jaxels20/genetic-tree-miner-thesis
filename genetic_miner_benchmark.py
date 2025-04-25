from src.FileLoader import FileLoader
from src.Evaluator import MultiEvaluator
from src.Mutator import Mutator, TournamentMutator
from src.Objective import Objective
from src.RandomTreeGenerator import BottomUpRandomBinaryGenerator, FootprintGuidedSequentialGenerator, InductiveNoiseInjectionGenerator, InductiveMinerGenerator
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
            mutator=TournamentMutator(random_creation_rate=0.2, elite_rate=0.2, tournament_size=0.4),
            generator=BottomUpRandomBinaryGenerator(),
            objective=Objective(metric_weights={"simplicity": 20, "refined_simplicity": 20, "ftr_fitness": 100, "ftr_precision": 50}),
            percentage_of_log=0.1,
            population_size=5,
            # stagnation_limit=10
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

    multi_evaluator = MultiEvaluator(eventlogs, methods_dict)
    results_df = multi_evaluator.evaluate_all({"simplicity": 20, "refined_simplicity": 20, "ftr_fitness": 100, "ftr_precision": 50})
    results_df.to_csv(OUTPUT_DIR + "results.csv", index=False)
    multi_evaluator.save_df_to_pdf(results_df, OUTPUT_DIR + "results.pdf")
    multi_evaluator.export_petri_nets(OUTPUT_DIR)
    multi_evaluator.plot_monitor_data()