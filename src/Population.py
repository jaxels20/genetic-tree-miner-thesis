
from typing import List
from ProcessTree import ProcessTree, Operator
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
        self.trees.append(deepcopy(tree))
        
    def add_trees(self, trees: List[ProcessTree]):
        """
        Adds multiple trees to the population
        """
        self.trees.extend(deepcopy(trees))
    
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
        return self.get_elite(1)[0]
    
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
    
    def is_equal(self, other: 'Population') -> bool:
        """
        Returns True if the two populations are equal without considering the order of the trees
        """
        if len(self.trees) != len(other.trees):
            return False
        
        for tree in self.trees:
            for other_tree in other.trees:
                if tree.is_equal(other_tree):
                    break
            else:
                return False
        return True
    
    def __str__(self) -> str:
        if not self.trees:
            return "Population is empty"

        max_tree_len = max(len(str(tree)) for tree in self.trees)  # Get longest tree representation
        max_fitness_len = max(len(f"{tree.get_fitness():.4f}") for tree in self.trees)  # Align fitness values

        result = f"Population of size {len(self.trees)}\n"
        for tree in self.trees:
            tree_str = str(tree).ljust(max_tree_len)  # Ensure uniform width for tree representation
            fitness_str = f"Fitness: {tree.get_fitness():.4f}".rjust(max_fitness_len + 9)  # Align fitness values
            result += f"{tree_str}  {fitness_str}\n"
        
        return result

if __name__ == "__main__":
    pop1 = Population([ProcessTree(operator=Operator.SEQUENCE), ProcessTree(operator=Operator.PARALLEL)])
    pop2 = Population([ProcessTree(operator=Operator.PARALLEL), ProcessTree(operator=Operator.SEQUENCE),])
    
    # check if two empty populations are equal
    print(pop1.is_equal(pop2))  # True