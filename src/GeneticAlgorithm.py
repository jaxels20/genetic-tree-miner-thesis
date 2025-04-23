from src.Objective import Objective
from src.RandomTreeGenerator import BottomUpRandomBinaryGenerator, FootprintGuidedSequentialGenerator, InductiveNoiseInjectionGenerator, InductiveMinerGenerator
from src.ProcessTree import ProcessTree
from src.EventLog import EventLog
from src.Mutator import Mutator, TournamentMutator
from src.Population import Population
from src.Monitor import Monitor
from src.Filtering import Filtering
import tqdm
import time
from typing import Union
import os
import time
class GeneticAlgorithm:
    def __init__(self, method_name):
        """
        :param min_fitness: Minimum fitness level to stop the algorithm
        :param max_generations: Maximum number of generations
        :param stagnation_limit: Maximum generations without improvement
        """
        self.method_name = method_name
        self.stagnation_counter = 0
        self.start_time = None
        self.best_tree = None
        self.monitor = Monitor()
        
    def _check_stopping_criteria(self, generation: int, population: Population, stagnation_limit: int, time_limit: int, min_fitness: float) -> bool:
        # Update the best tree
        best_tree_b_update = self.best_tree.get_fitness() if self.best_tree is not None else None
        self._update_best_tree(population)
        
        # Criterion 1: Minimum fitness level reached
        if min_fitness is not None:
            if self.best_tree >= min_fitness:
                print(f"Minimum fitness level reached in generation {generation}")
                return True
        
        # Criterion 2: No improvement for `stagnation_limit` generations
        if stagnation_limit is not None and best_tree_b_update is not None:
            if self.best_tree.get_fitness() > best_tree_b_update:
                self.stagnation_counter = 0  # Reset stagnation counter
            else:
                self.stagnation_counter += 1
                if self.stagnation_counter >= stagnation_limit:
                    print(f"Stagnation limit reached in generation {generation}")
                    return True
            
        # Criterion 3: Time limit reached
        if time_limit is not None:                
            if time.time() - self.start_time >= time_limit:
                print(f"Time limit reached in generation {generation}")
                return True
        
        return False
    
    def _update_best_tree(self, population: Population):
        best_tree = population.get_best_tree()
        if self.best_tree is None or best_tree.get_fitness() > self.best_tree.get_fitness():
            self.best_tree = best_tree
    
    def run(self,
            eventlog: EventLog,
            population_size: int,
            mutator: Union[Mutator, TournamentMutator], 
            generator: Union[BottomUpRandomBinaryGenerator, FootprintGuidedSequentialGenerator, InductiveNoiseInjectionGenerator, InductiveMinerGenerator],
            objective: Objective,
            percentage_of_log: float,
            max_generations: int,
            min_fitness: float,
            stagnation_limit: int,
            time_limit: int, # Time limit in seconds
            export_monitor_path: str,
        ) -> ProcessTree:
        # Start the timer
        self.start_time = time.time()
        
        # Filter the log
        eventlog = Filtering.filter_eventlog_by_top_percentage_unique(eventlog, percentage_of_log, True)
        objective.set_event_log(eventlog)
        mutator.set_event_log(eventlog)
        
        # Generate initial population
        if isinstance(generator, BottomUpRandomBinaryGenerator):
            population = generator.generate_population(eventlog.unique_activities(), n=population_size)
        elif isinstance(generator, FootprintGuidedSequentialGenerator):
            population = generator.generate_population(eventlog, n=population_size)
        elif isinstance(generator, InductiveNoiseInjectionGenerator):
            population = generator.generate_population(eventlog, n=population_size)
        elif isinstance(generator, InductiveMinerGenerator):
            population = generator.generate_population(eventlog, n=population_size)
        else:
            raise ValueError("Invalid generator type. Must be one of: BottomUpRandomBinaryGenerator, FootprintGuidedSequentialGenerator, InductiveNoiseInjectionGenerator, InductiveMinerGenerator.")
        
        
        for generation in tqdm.tqdm(range(max_generations), desc="Discovering process tree", unit="generation"):   
            # Evaluate the fitness of each tree
            objective.evaluate_population(population)
            
            # Observe the population
            self.monitor.observe(generation, population)
            
            # check stopping criteria
            stop = self._check_stopping_criteria(generation, population, stagnation_limit, time_limit, min_fitness)
            if stop:
                break
               
            # Generate a new population
            population = mutator.generate_new_population(population)
        
        if export_monitor_path is not None:
            self.monitor.save_objective_results(export_monitor_path, eventlog.name, self.method_name)
        
        return self.best_tree
    