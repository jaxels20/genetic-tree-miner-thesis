from Objective import SimpleWeightedAverage
from RandomTreeGenerator import BottomUpBinaryTreeGenerator
from ProcessTree import ProcessTree
from EventLog import EventLog
from Mutator import Mutator

# Remove all prints from the pm4py library
import logging
logging.getLogger('pm4py').setLevel(logging.CRITICAL)


class GeneticAlgorithm:
    def __init__(self, min_fitness=0.95, max_generations=100, stagnation_limit=10):
        """
        :param min_fitness: Minimum fitness level to stop the algorithm
        :param max_generations: Maximum number of generations
        :param stagnation_limit: Maximum generations without improvement
        """
        self.min_fitness = min_fitness
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit
    
    def run(self, eventlog: EventLog) -> ProcessTree:
        population_size = 100
        survival_rate = 0.5
        
        # Initialize the population
        generator = BottomUpBinaryTreeGenerator()
        population = generator.generate_population(eventlog.unique_activities(), population_size)
        
        best_fitness = 0
        best_tree = None
        stagnation_counter = 0
        
        for generation in range(self.max_generations):
            # Evaluate the fitness of each tree
            for tree in population:
                obj = SimpleWeightedAverage(tree, eventlog)
                fitness = obj.fitness()
                tree.set_fitness(fitness)
            
            # Select the best trees
            num_survivors = int(population_size * survival_rate)
            selected_trees = sorted(population, key=lambda tree: tree.get_fitness(), reverse=True)[:num_survivors]
            
            # Check stopping criteria
            current_best_tree = selected_trees[0]
            current_best_fitness = current_best_tree.get_fitness()
            
            # Criterion 1: Minimum fitness level reached
            if current_best_fitness >= self.min_fitness:
                return current_best_tree
            
            # Criterion 2: No improvement for `stagnation_limit` generations
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_tree = current_best_tree
                stagnation_counter = 0  # Reset stagnation counter
            else:
                stagnation_counter += 1
                if stagnation_counter >= self.stagnation_limit:
                    return best_tree
            
            # Generate a new population
            mutator = Mutator(eventlog)
            population = mutator.generate_new_population(selected_trees, population_size)
        
        # Criterion 3: Maximum generations reached
        return best_tree if best_tree else max(population, key=lambda tree: tree.get_fitness())
    
if __name__ == "__main__":
    eventlog = EventLog.from_trace_list(["AB", "AC"])
    ga = GeneticAlgorithm(min_fitness=0.90, max_generations=100, stagnation_limit=15)
    best_tree = ga.run(eventlog=eventlog)
    print(best_tree)
    
""" from Objective import SimpleWeightedAverage
from RandomTreeGenerator import BottomUpBinaryTreeGenerator
from ProcessTree import ProcessTree
from EventLog import EventLog
from Mutator import Mutator

# Remove all prints from the pm4py library
import logging
logging.getLogger('pm4py').setLevel(logging.CRITICAL)


class GeneticAlgorithm:
    def __init__(self):
        pass
    
    
    def run(self, eventlog: EventLog) -> ProcessTree:
        num_generations = 100
        population_size = 100
        survival_rate = 0.5
        
        # Initialize the population
        generator = BottomUpBinaryTreeGenerator()
        population = generator.generate_population(eventlog.unique_activities(), population_size)
        
        for i in range(num_generations):
            # Evaluate the fitness of each tree
            for tree in population:
                obj = SimpleWeightedAverage(tree, eventlog)
                fitness = obj.fitness()
                tree.set_fitness(fitness)
            
            # Select the best trees
            num_survivors = int(population_size * survival_rate)
            selected_trees = sorted(population, key=lambda tree: tree.get_fitness(), reverse=True)[:num_survivors]
            
            # Generate a new population
            mutator = Mutator(eventlog)
            population = mutator.generate_new_population(selected_trees, population_size)

        # Return the best tree from the final generation
        for tree in population:
            obj = SimpleWeightedAverage(tree, eventlog)
            fitness = obj.fitness()
            tree.set_fitness(fitness)
        best_tree = max(population)        
        return best_tree
    
if __name__ == "__main__":
    eventlog = EventLog.from_trace_list(["AB", "AC"])
    ga = GeneticAlgorithm()
    best_tree = ga.run(eventlog=eventlog)
    print(best_tree) """