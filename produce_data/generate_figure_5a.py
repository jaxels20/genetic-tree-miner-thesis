
# add the parent directory to the system path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import optuna
from src.Discovery import Discovery
from src.FileLoader import FileLoader
from src.Evaluator import SingleEvaluator
from src.Mutator import TournamentMutator
from src.RandomTreeGenerator import InductiveNoiseInjectionGenerator
from src.Objective import Objective
import pandas as pd
import multiprocessing

TIME_LIMIT = 5*60
STAGNATION_LIMIT = 50
PERCENTAGE_OF_LOG = 0.05
OPTUNA_TIMEOUT_LIMIT = 60*60*18
INPUT_DIR = "./real_life_datasets/"
OUTPUT_DIR = "./data/figure_5a/" 

FITNESS_WEIGHTS = {
    "simplicity": 10,
    "refined_simplicity": 10,
    "ftr_fitness": 50,
    "ftr_precision": 30
}

def objective(trial, event_log, fitness_weights=dict[str, float]):
    # Suggest hyperparameters
    random_creation_rate = trial.suggest_float("random_creation_rate", 0.0, 1.0)
    elite_rate = trial.suggest_float("elite_rate", 0.0, 1.0)
    population_size = trial.suggest_int("population_size", 20, 60)
    
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
        

    log_filtering = trial.suggest_float("log_filtering", 0.0, 0.1)
    generator = InductiveNoiseInjectionGenerator(log_filtering=log_filtering)

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
            stagnation_limit=STAGNATION_LIMIT,
            time_limit=TIME_LIMIT,
        )

        # Evaluate fitness — should return a single value (higher is better)
        evaluator = SingleEvaluator(petri_net, event_log)
        fitness_score = evaluator.get_objective_fitness(fitness_weights)

        return fitness_score
    except Exception as e:
        print(f"Error during discovery: {e}")
        return 0.0


def optimize_dataset(dataset):
    loader = FileLoader()
    data = []

    print(f"Loading dataset {dataset}...")

    eventlog = loader.load_eventlog(f"{INPUT_DIR}/{dataset}")

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )
    
    study.optimize(
        lambda trial: objective(trial, eventlog, FITNESS_WEIGHTS),
        show_progress_bar=False,
        n_trials=None,
        timeout=OPTUNA_TIMEOUT_LIMIT
    )

    best_params = study.best_params
    best_value = study.best_value
    data.append({**best_params, "objective": best_value, "dataset": f"{dataset.split('.')[0]}"})

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":    
    datasets = os.listdir(INPUT_DIR)
    with multiprocessing.Pool(processes=min(len(datasets), multiprocessing.cpu_count())) as pool:
        results = pool.map(optimize_dataset, datasets) 

    combined_df = pd.concat(results, ignore_index=True)  # Combine all individual DataFrames
    
    # check if the output directory exists, if not create it
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(os.path.join(OUTPUT_DIR, "figure_5a.csv"), index=False)        