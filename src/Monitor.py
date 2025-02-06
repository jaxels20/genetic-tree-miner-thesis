from Population import Population
import matplotlib.pyplot as plt

class Monitor:
    def __init__(self):
        self.generations = []
        self.populations = []
        
        self.best_trees = []
        self.best_fitnesses = []
        
        
        
    def observe(self, generation: int, population: Population):
        """
        Observe the population and the generation
        """
        self.generations.append(generation)
        self.populations.append(population)
        
        best_tree = population.get_best_tree()
        self.best_trees.append(best_tree)
        self.best_fitnesses.append(best_tree.get_fitness())
    
    def plot_fitness(self) -> None:
        plt.plot(self.generations, self.best_fitnesses)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness over generations")
        plt.show()

    
    def plot_population_size(self):
        population_sizes = [len(population) for population in self.populations]
        plt.plot(self.generations, population_sizes)
        plt.xlabel("Generation")
        plt.ylabel("Population size")
        plt.title("Population size over generations")
        plt.show()
    
    def print_best_trees(self):
        for i, tree in enumerate(self.best_trees):
            print(f"Generation {i}: {tree} Fitness: {tree.get_fitness()}")