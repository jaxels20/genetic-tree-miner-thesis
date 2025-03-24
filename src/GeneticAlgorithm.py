from src.Objective import SimpleWeightedScore
from src.RandomTreeGenerator import BottomUpBinaryTreeGenerator
from src.RandomTreeGenerator import SequentialTreeGenerator
from src.ProcessTree import ProcessTree
from src.EventLog import EventLog
from src.Mutator import Mutator
from src.Population import Population
from src.Monitor import Monitor
from src.ProcessTreeRegister import ProcessTreeRegister
import tqdm
import time
from pprint import pprint


class GeneticAlgorithm:
    def __init__(self, mutator: Mutator, min_fitness: float = None, max_generations: int = 100, stagnation_limit = None, time_limit = None, population_size = 100):
        """
        :param min_fitness: Minimum fitness level to stop the algorithm
        :param max_generations: Maximum number of generations
        :param stagnation_limit: Maximum generations without improvement
        """
        self.min_fitness = min_fitness
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit
        self.stagnation_counter = 0
        self.best_tree = None
        self.time_limit = time_limit # Time limit in seconds
        self.start_time = None
        self.population_size = population_size

        self.mutator = mutator
        self.monitor = Monitor()
        self.process_tree_register = ProcessTreeRegister({})
        
    def _check_stopping_criteria(self, generation: int, population: Population) -> bool:
        # Update the best tree
        best_tree_b_update = self.best_tree.get_fitness() if self.best_tree is not None else None
        self._update_best_tree(population)
        
        # Criterion 1: Minimum fitness level reached
        if self.min_fitness is not None:
            if self.best_tree >= self.min_fitness:
                print(f"Minimum fitness level reached in generation {generation}")
                return True
        
        # Criterion 2: No improvement for `stagnation_limit` generations
        if self.stagnation_limit is not None and best_tree_b_update is not None:
            if self.best_tree.get_fitness() > best_tree_b_update:
                self.stagnation_counter = 0  # Reset stagnation counter
            else:
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.stagnation_limit:
                    print(f"Stagnation limit reached in generation {generation}")
                    return True
            
        # Criterion 3: Time limit reached
        if self.time_limit is not None:                
            if time.time() - self.start_time >= self.time_limit:
                print(f"Time limit reached in generation {generation}")
                return True
        
        return False
    
    def _update_best_tree(self, population: Population):
        best_tree = population.get_best_tree()
        if self.best_tree is None or best_tree.get_fitness() > self.best_tree.get_fitness():
            self.best_tree = best_tree
    
    def run(self, eventlog: EventLog) -> ProcessTree:
        # Start the timer
        self.start_time = time.time()
        
        # Initialize the population
        generator = SequentialTreeGenerator()
        population = generator.generate_population(eventlog, n=self.population_size)
        
        for generation in tqdm.tqdm(range(self.max_generations), desc="Discovering process tree", unit="generation"):
            # Evaluate the fitness of each tree
            SimpleWeightedScore.evaluate_population(population, eventlog, self.process_tree_register, num_processes=1)
            
            # Observe the population
            self.monitor.observe(generation, population)
            
            # check stopping criteria
            stop = self._check_stopping_criteria(generation, population)
            if stop:
                break
               
            # Generate a new population
            population = self.mutator.generate_new_population(population)
            # population = self.mutator.tournament_population_generation(population)
        
        return self.best_tree
    