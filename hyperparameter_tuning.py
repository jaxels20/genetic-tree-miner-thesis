import optuna
import os
import time
import pickle as pkl
from optuna_dashboard import run_server
from src.Discovery import Discovery
from src.FileLoader import FileLoader
from src.Evaluator import SingleEvaluator
from src.Mutator import TournamentMutator, Mutator
from src.RandomTreeGenerator import BottomUpBinaryTreeGenerator, SequentialTreeGenerator, InjectionTreeGenerator
from src.Objective import Objective
import pandas as pd


FITNESS_WEIGHTS = {
    "simplicity": 20,
    "refined_simplicity": 20,
    "ftr_fitness": 100,
    "ftr_precision": 50
}
INPUT_DIR = "./real_life_datasets/"
OUTPUT_DIR = "./real_life_datasets_results/" 


def objective(trial, event_log, fitness_weights=dict[str, float]):
    # Suggest hyperparameters
    random_creation_rate = trial.suggest_float("random_creation_rate", 0.0, 1.0)
    mutation_rate = trial.suggest_float("mutation_rate", 0.0, 1.0)
    crossover_rate = trial.suggest_float("crossover_rate", 0.0, 1.0)
    elite_rate = trial.suggest_float("elite_rate", 0.0, 1.0)
    population_size = trial.suggest_int("population_size", 20, 100)
    percentage_of_log = trial.suggest_float("percentage_of_log", 0.01, 0.5)
    generator = trial.suggest_categorical("generator", ["Buttom", "Sequential"])
    mutator = trial.suggest_categorical("mutator", ["Tournament", "NonTournament"])
    
    # Normalize them so that they sum to 1
    total = mutation_rate + crossover_rate + random_creation_rate + elite_rate
    mutation_rate = mutation_rate / total
    crossover_rate = crossover_rate / total
    random_creation_rate = random_creation_rate / total
    elite_rate = elite_rate / total

    # Create the mutator
    if mutator == "Tournament":
        tournament_size = trial.suggest_float("tournament_size", 0.3, 0.5)
        mutator = TournamentMutator(
                    random_creation_rate=random_creation_rate, 
                    crossover_rate=crossover_rate, 
                    mutation_rate=mutation_rate, 
                    elite_rate=elite_rate,
                    tournament_size=tournament_size)
    elif mutator == "NonTournament":
        mutator = Mutator(
                    random_creation_rate=random_creation_rate, 
                    crossover_rate=crossover_rate, 
                    mutation_rate=mutation_rate, 
                    elite_rate=elite_rate)
        
    # Create the generator
    if generator == "Buttom":
        generator = BottomUpBinaryTreeGenerator()
    elif generator == "Sequential":
        generator = SequentialTreeGenerator()
    elif generator == "Injection":
        log_filtering = trial.suggest_float("log_filtering", 0.0, 1.0)
        generator = InjectionTreeGenerator(log_filtering=log_filtering)

    # Run the genetic miner
    start = time.time()
    petri_net = Discovery.genetic_algorithm(
        event_log,
        method_name="Genetic Miner",
        objective=Objective(metric_weights={"simplicity": 20, "refined_simplicity": 20, "ftr_fitness": 100, "ftr_precision": 50}),
        mutator=mutator,
        generator=generator,
        percentage_of_log=percentage_of_log,
        population_size=population_size,
        stagnation_limit=15,
        time_limit=10,
    )
    end = time.time()

    # Evaluate fitness — should return a single value (higher is better)
    evaluator = SingleEvaluator(petri_net, event_log)
    fitness_score = evaluator.get_objective_fitness(fitness_weights)

    return fitness_score


if __name__ == "__main__":    
    data = []
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()

    for dataset_dir in dataset_dirs:
        data = []

        # Assume only one file per directory
        if dataset_dir != "BPI_Challenge_2013_open_problems" and dataset_dir != "BPI_Challenge_2013_closed_problems":
            continue
        
        print(f"Loading dataset {dataset_dir}...")
        
        xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
        if len(xes_file) == 0:
            continue
        elif len(xes_file) == 1:
            loaded_log = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")
            eventlog = loaded_log
        else:
            raise ValueError("More than one xes file in the directory")

        for i in range(2):
            sampler = optuna.samplers.TPESampler()
            study = optuna.create_study(direction="maximize", sampler=sampler, storage="sqlite:///db.sqlite3")
            study.optimize(
                lambda trial: objective(trial, eventlog, FITNESS_WEIGHTS),
                show_progress_bar=True, 
                n_trials=None,
                timeout=600,
                n_jobs=1,
            )
                        
            best_params = study.best_params
            best_value = study.best_value
            
            data.append(
                {
                    **best_params,
                    "objective": best_value,
                    "dataset": dataset_dir + f"_{i}",
                }
            )
        
        df = pd.DataFrame(data)
        
        
        df.to_csv(f"./best_params/{dataset_dir}.csv", index=False)

        

