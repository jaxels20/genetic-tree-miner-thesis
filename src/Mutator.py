import random
from typing import List
from ProcessTree import ProcessTree, Operator
from RandomTreeGenerator import BottomUpBinaryTreeGenerator
from EventLog import EventLog
from copy import deepcopy
from Population import Population

# TODO ONLY Create Valid trees and onyl create trees with all activities in the log

class MutatorBase:
    def __init__(self, EventLog: EventLog):
        self.EventLog = EventLog

    def generate_new_population(self, old_population: List[ProcessTree], new_population_size: int) -> List[ProcessTree]:
        raise NotImplementedError

class Mutator(MutatorBase):
    def __init__(self, EventLog: EventLog, random_creation_rate: float, crossover_rate: float, mutation_rate: float, elite_rate: float):
        super().__init__(EventLog)
        self.random_creation_rate = random_creation_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate 
        
    def random_creation(self, num_new_trees: int) -> List[ProcessTree]:
        generator = BottomUpBinaryTreeGenerator()
        new_trees = generator.generate_population(self.EventLog.unique_activities(), num_new_trees)
        return new_trees.get_population()
    
    def crossover(self, parent1: ProcessTree, parent2: ProcessTree) -> ProcessTree:
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
        
        if random.random() > 0.5:
            new_parent1.if_missing_insert_activities(self.EventLog.unique_activities())
            return new_parent1
        else:
            new_parent2.if_missing_insert_activities(self.EventLog.unique_activities())
            return new_parent2

    def mutation(self, process_tree: ProcessTree) -> ProcessTree:
        
        def node_mutation(tree):
            nodes = tree.get_all_nodes()
            if not nodes:
                raise ValueError("Tree has no nodes")
            
            node = random.choice(nodes)
            if node.operator:
                node.operator = random.choice([Operator.SEQUENCE, Operator.XOR, Operator.PARALLEL, Operator.LOOP])
                random.shuffle(node.children)
            elif node.label:
                node.label = random.choice(list(self.EventLog.unique_activities()))
            return tree
                  
        def subtree_removal(tree, max_attempts=10):
            nodes = tree.get_all_nodes()
            if len(nodes) <= 1:
                return tree  # Cannot remove from a single-node tree
            
            attempts = 0
            while attempts < max_attempts:
                node = random.choice(nodes)
                if node.parent:
                    node.parent.children.remove(node)  # Remove node from its parent

                    if tree.is_valid():
                        return tree  # Return immediately if valid
                    
                    # If invalid, restore the node
                    node.parent.children.append(node)
                
                attempts += 1
            
            return tree  # Return the original tree if no valid removal was found
        
        def node_addition(tree):
            operator_nodes = [node for node in tree.get_all_nodes() if node.operator]
            if not operator_nodes:
                raise ValueError("Tree has no operator nodes")
            
            parent = random.choice(operator_nodes)
            new_leaf = ProcessTree(label=random.choice(list(self.EventLog.unique_activities())))
            parent.add_child(new_leaf)
            return tree

        mutation_type = random.choice(['node_mutation', 'node_addition'])

        if mutation_type == 'node_mutation':
            new_tree = node_mutation(process_tree)
            if not new_tree.is_valid():
                raise ValueError("Invalid tree (node_mutation)") 
        # elif mutation_type == 'subtree_removal':
        #     new_tree = subtree_removal(process_tree)
        #     if not new_tree.is_valid():
        #         raise ValueError("Invalid tree(subtree_removal)")
        elif mutation_type == 'node_addition':
            new_tree = node_addition(process_tree)
            if not new_tree.is_valid():
                raise ValueError("Invalid tree(node_addition)")
        
        new_tree.if_missing_insert_activities(self.EventLog.unique_activities())
        return new_tree
    
    def generate_new_population(self, old_population: Population) -> Population:
        new_population = Population([])

        # Add elite trees
        elite = old_population.get_elite(int(len(old_population) * self.elite_rate))
        new_population.add_trees(elite)

        # Add random trees
        random_count = int(len(old_population) * self.random_creation_rate)
        new_population.add_trees(self.random_creation(random_count))
        
        # Add crossover trees
        crossover_count = int(len(old_population) * self.crossover_rate)
        for _ in range(crossover_count):
            parent1, parent2 = random.sample(old_population.get_population(), 2)
            new_population.add_tree(self.crossover(parent1, parent2))
                
        # Add mutation trees
        mutation_count = int(len(old_population) * self.mutation_rate)
        for _ in range(mutation_count):
            parent = random.choice(old_population)
            new_population.add_tree(self.mutation(parent))
                    
        # check if the tree is strictly valid if not remove it and insert random new tree     
        num_removed = new_population.ensure_strictly_valid(self.EventLog.unique_activities())
        
        if num_removed > 0:
            new_population.add_trees(self.random_creation(num_removed))
    
        return new_population

