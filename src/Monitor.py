from src.Population import Population
import matplotlib.pyplot as plt
import pickle
import os

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
    
    def save_objective_results(self, dataset_name, method_name) -> None:
        result_dict = {}
        for generation, best_tree_fitness in zip(self.generations, self.best_fitnesses):
            result_dict[generation] = best_tree_fitness
        
        os.makedirs(f"./monitor_data/data/{dataset_name}", exist_ok=True)
        with open(f"./monitor_data/data/{dataset_name}/{method_name}.pkl", "wb") as f:
            pickle.dump((dataset_name, method_name, result_dict), f)       
    
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
        max_gen_len = len(f"Generation {len(self.best_trees) - 1}")  # Get width of longest "Generation X"
        max_tree_len = max(len(str(tree)) for tree in self.best_trees)  # Get longest tree representation
        max_fitness_len = max(len(f"{tree.get_fitness():.4f}") for tree in self.best_trees)  # Align fitness values

        for i, tree in enumerate(self.best_trees):
            gen_str = f"Generation {i}".ljust(max_gen_len)  # Ensure uniform width for generation
            tree_str = str(tree).ljust(max_tree_len)  # Ensure uniform width for tree representation
            fitness_str = f"Fitness: {tree.get_fitness():.4f}".rjust(max_fitness_len + 9)  # Align fitness values
            print(f"{gen_str}  {tree_str}  {fitness_str}")

    def plot_largest_tree_size(self):
        # tree_sizes = [max([len(str(tree)) for tree in population]) for population in self.populations]
        tree_sizes = [max([tree.get_size() for tree in population]) for population in self.populations]
        plt.plot(self.generations, tree_sizes)
        plt.xlabel("Generation")
        plt.ylabel("Largest tree size")
        plt.title("Largest tree size over generations")
        plt.show()
        
    def plot_size_of_best_tree(self):
        tree_sizes = [best_tree.get_size() for best_tree in self.best_trees]
        plt.plot(self.generations, tree_sizes)
        plt.xlabel("Generation")
        plt.ylabel("Size of best tree")
        plt.title("Size of best tree over generations")
        plt.show()
        
