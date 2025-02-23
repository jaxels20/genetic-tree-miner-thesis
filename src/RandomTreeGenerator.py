import random
from typing import List
from src.ProcessTree import ProcessTree, Operator
from src.Population import Population

# TODO: MAKE SURE THAT THE RANDOM TREES ARE VALID (E.G. NO OR OPERATOR WITH ONLY ONE CHILD)

class RandomTreeGeneratorBase:
    def __init__(self):
        pass
    
    def generate_population(self, unique_activities: List[str], n: int) -> List[ProcessTree]:
        raise NotImplementedError


class BottomUpBinaryTreeGenerator(RandomTreeGeneratorBase):
    def __init__(self):
        pass
    
    def _generate_naive_binary_tree(self, unique_activities: List[str]) -> ProcessTree:
        """
        Generates a random binary process tree using the given unique activities.
        
        The function works bottom-up by randomly pairing activities and assigning
        an operator from {SEQUENCE, XOR, PARALLEL, LOOP, OR}. The process continues until a
        single root node remains.
        """
        if not unique_activities:
            raise ValueError("The list of unique activities cannot be empty.")
            
        # Convert activities into leaf nodes
        nodes = [ProcessTree(label=activity) for activity in unique_activities]
        
        while len(nodes) > 1:
            random.shuffle(nodes)  # Shuffle to ensure randomness
            new_nodes = []
            
            for i in range(0, len(nodes) - 1, 2):
                operator = random.choice([Operator.SEQUENCE, Operator.XOR, Operator.PARALLEL, Operator.LOOP])
                parent = ProcessTree(operator=operator)
                parent.add_child(nodes[i])
                parent.add_child(nodes[i + 1])
                new_nodes.append(parent)
            
            # If odd number of nodes, pass the last one to the next round
            if len(nodes) % 2 == 1:
                new_nodes.append(nodes[-1])
            
            nodes = new_nodes  # Move up a level
        
        return nodes[0]  # Root of the tree

    def generate_population(self, unique_activities: List[str], n: int) -> Population:
        """
        Generates n unique random binary process trees using the given unique activities.
        """
        trees = []
        for _ in range(n):
            tree = self._generate_naive_binary_tree(unique_activities)
            trees.append(tree)

        population = Population(trees)
        return population
        
    
    
if __name__ == "__main__":
    unique_activities = ["A", "B", "C"]
    generator = BottomUpBinaryTreeGenerator()
    tree = generator.generate_population(unique_activities, 100)[0]
    print(tree)
