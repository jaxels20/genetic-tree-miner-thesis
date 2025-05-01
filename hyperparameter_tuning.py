import optuna
import os
import time
import pickle as pkl
from src.Discovery import Discovery
from src.FileLoader import FileLoader
from src.Evaluator import SingleEvaluator
from src.Mutator import TournamentMutator, Mutator
from src.RandomTreeGenerator import BottomUpRandomBinaryGenerator, FootprintGuidedSequentialGenerator, InductiveNoiseInjectionGenerator, InductiveMinerGenerator
from src.Objective import Objective
import pandas as pd
import multiprocessing


FITNESS_WEIGHTS = {
    "simplicity": 10,
    "refined_simplicity": 10,
    "ftr_fitness": 50,
    "ftr_precision": 30
}
PERCENTAGE_OF_LOG = 0.05
MUTATOR = "Tournament"
INPUT_DIR = "./real_life_datasets/"
OUTPUT_DIR = "./real_life_datasets_results/" 


def objective(trial, event_log, fitness_weights=dict[str, float]):
    # Suggest hyperparameters
    random_creation_rate = trial.suggest_float("random_creation_rate", 0.0, 1.0)
    elite_rate = trial.suggest_float("elite_rate", 0.0, 1.0)
    population_size = trial.suggest_int("population_size", 20, 60)
    generator = trial.suggest_categorical("generator", ["BottomUpRandomBinaryGenerator", "FootprintGuidedSequentialGenerator", "InductiveNoiseInjectionGenerator", "InductiveMinerGenerator"])
    
    # Create the mutator
    tournament_size = trial.suggest_float("tournament_size", 0.1, 0.3)
    tournament_rate = trial.suggest_float("tournament_rate", 0.0, 1.0)
    tournament_mutation_rate = trial.suggest_float("tournament_mutation_rate", 0.0, 1.0)
    
    total = random_creation_rate + elite_rate + tournament_rate
    random_creation_rate = random_creation_rate / total
    elite_rate = elite_rate / total
    tournament_rate = tournament_rate / total
    
    mutator = TournamentMutator(
                random_creation_rate=random_creation_rate,  
                elite_rate=elite_rate,
                tournament_rate=tournament_rate,
                tournament_size=tournament_size,
                tournament_mutation_rate=tournament_mutation_rate
    )
        
    # Create the generator
    if generator == "BottomUpRandomBinaryGenerator":
        generator = BottomUpRandomBinaryGenerator()
    elif generator == "FootprintGuidedSequentialGenerator":
        generator = FootprintGuidedSequentialGenerator()
    elif generator == "InductiveNoiseInjectionGenerator":
        log_filtering = trial.suggest_float("log_filtering", 0.0, 0.1)
        generator = InductiveNoiseInjectionGenerator(log_filtering=log_filtering)
    elif generator == "InductiveMinerGenerator":
        generator = InductiveMinerGenerator()

    # Run the genetic miner
    try:
        petri_net = Discovery.genetic_algorithm(
            event_log,
            method_name="Genetic Miner",
            objective=Objective(metric_weights=FITNESS_WEIGHTS),
            mutator=mutator,
            generator=generator,
            percentage_of_log=PERCENTAGE_OF_LOG,
            population_size=population_size,
            stagnation_limit=50,
            time_limit=5 * 60,
        )

        # Evaluate fitness — should return a single value (higher is better)
        evaluator = SingleEvaluator(petri_net, event_log)
        fitness_score = evaluator.get_objective_fitness(fitness_weights)

        return fitness_score
    except Exception as e:
        print(f"Error during discovery: {e}")
        return 0.0


def optimize_dataset(dataset_dir):
    loader = FileLoader()
    data = []

    print(f"Loading dataset {dataset_dir}...")

    xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
    if len(xes_file) != 1:
        print(f"Skipping {dataset_dir}, expected one .xes file, found {len(xes_file)}.")
        return

    eventlog = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")

    for i in range(1):
        sampler = optuna.samplers.TPESampler(seed=i)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            storage=f"sqlite:///best_params/db{dataset_dir}.sqlite3",
        )
        study.optimize(
            lambda trial: objective(trial, eventlog, FITNESS_WEIGHTS),
            show_progress_bar=False,
            n_trials=None,
            timeout=60 * 60 * 18
        )

        best_params = study.best_params
        best_value = study.best_value
        data.append({**best_params, "objective": best_value, "dataset": f"{dataset_dir}_{i}"})

    df = pd.DataFrame(data)
    os.makedirs("./best_params", exist_ok=True)
    df.to_csv(f"./best_params/{dataset_dir}.csv", index=False)
    print(f"Finished {dataset_dir}")


if __name__ == "__main__":    
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    with multiprocessing.Pool(processes=min(len(dataset_dirs), multiprocessing.cpu_count())) as pool:
        pool.map(optimize_dataset, dataset_dirs)
        

