from Objective import SimpleWeightedAverage
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
    print(best_tree)