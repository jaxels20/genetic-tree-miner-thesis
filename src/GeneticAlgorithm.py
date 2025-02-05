from Objective import Objective
from RandomTreeGenerator import RandomTreeGenerator
from ProcessTree import ProcessTree
from EventLog import EventLog
from Mutator import Mutator

class GeneticAlgorithm:
    def __init__(self):
        pass
    
    
    def run(self):
        num_generations = 100
        population_size = 100
        survival_rate = 0.1
        eventlog = EventLog.from_trace_list(["ABBC", "ABBBC", "ABBBBC"])
        
        # Initialize the population
        generator = RandomTreeGenerator()
        population = generator.generate_naive_binary_trees(eventlog.unique_activities(), population_size)
        
        for i in range(num_generations):
            # Evaluate the fitness of each tree
            for tree in population:
                obj = Objective(tree, eventlog)
                fitness = obj.fitness()
                tree.set_fitness(fitness)
            
            # Select the best trees
            num_survivors = int(population_size * survival_rate)
            selected_trees = sorted(population, key=lambda tree: tree.get_fitness(), reverse=True)[:num_survivors]
            
            # Generate a new population
            mutator = Mutator()
            population = mutator.generate_new_population(selected_trees, population_size)

        # Return the best tree from the final generation
        for tree in population:
            obj = Objective(tree, eventlog)
            fitness = obj.fitness()
            tree.set_fitness(fitness)
        best_tree = max(population)        
        return best_tree
    
if __name__ == "__main__":
    ga = GeneticAlgorithm()
    best_tree = ga.run()
    print(best_tree)