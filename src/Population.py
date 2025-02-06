
from typing import List
from ProcessTree import ProcessTree
from EventLog import EventLog
from copy import deepcopy

class Population:
    def __init__(self, trees: List[ProcessTree]):
        self.trees = trees

    def get_population(self) -> List[ProcessTree]:
        """
        Returns the population
        """
        return self.trees
    
    def add_tree(self, tree: ProcessTree):
        """
        Adds a tree to the population
        """
        self.trees.append(tree)
        
    def add_trees(self, trees: List[ProcessTree]):
        """
        Adds multiple trees to the population
        """
        self.trees.extend(trees)
    
    def get_elite(self, num_elite: int) -> List[ProcessTree]:
        """
        Returns the top `num_elite` of the population
        """
        # check if the number of elite trees is greater than the population size
        if num_elite > len(self.trees):
            raise ValueError("Number of elite trees is greater than the population size")
        
        return deepcopy(sorted(self.trees, key=lambda tree: tree.get_fitness(), reverse=True)[:num_elite])
    
    def get_best_tree(self) -> ProcessTree:
        """
        Returns the best tree in the population
        """
        return deepcopy(max(self.trees, key=lambda tree: tree.get_fitness()))
    
    def get_worst_tree(self) -> ProcessTree:
        """
        Returns the worst tree in the population
        """
        return deepcopy(min(self.trees, key=lambda tree: tree.get_fitness()))
    
    def get_average_fitness_of_elite(self, survival_rate: float) -> float:
        """
        Returns the average fitness of the elite trees
        """
        elite = self.get_elite(survival_rate)
        return sum(tree.get_fitness() for tree in elite) / len(elite)
    
    def __getitem__(self, index: int) -> ProcessTree:
        return self.trees[index]
    
    def __len__(self) -> int:
        return len(self.trees)