import random
from typing import List
from ProcessTree import ProcessTree

class Mutator:
    def __init__(self):
        pass


    @staticmethod
    def generate_new_population(selected_trees: List[ProcessTree], X: int) -> List[ProcessTree]:
        """
        Given a list of selected process trees, generates a new population of size X using crossover.
        - selected_trees: List of best trees from the previous generation.
        - X: Desired size of the new population.
        - Returns a new list of process trees.
        """
        
        def crossover(parent1: ProcessTree, parent2: ProcessTree) -> ProcessTree:
            """
            Performs subtree crossover between two process trees.
            - Randomly selects a subtree in parent1 and replaces it with a subtree from parent2.
            - If either tree is too small, returns a shallow copy of parent1 or parent2.
            """
            # Base case: If one of the parents is a leaf node, return a copy of it
            if not parent1.children or not parent2.children:
                return ProcessTree(operator=parent1.operator, label=parent1.label)

            # Randomly select crossover points (subtrees)
            subtree1 = random.choice(parent1.children)
            subtree2 = random.choice(parent2.children)

            # Create copies of the parents
            new_parent1 = ProcessTree(operator=parent1.operator, label=parent1.label, children=[c for c in parent1.children])
            new_parent2 = ProcessTree(operator=parent2.operator, label=parent2.label, children=[c for c in parent2.children])

            # Perform subtree swap
            idx1 = new_parent1.children.index(subtree1)
            idx2 = new_parent2.children.index(subtree2)
            new_parent1.children[idx1] = subtree2
            new_parent2.children[idx2] = subtree1

            return new_parent1 if random.random() < 0.5 else new_parent2
        
        new_population = []
        while len(new_population) < X:
            # Randomly select two parents from the existing population
            parent1, parent2 = random.sample(selected_trees, 2)

            # Perform crossover and add offspring to the new population
            offspring = crossover(parent1, parent2)
            new_population.append(offspring)

        return new_population


