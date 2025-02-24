import random
from typing import List
from src.ProcessTree import ProcessTree, Operator
from src.RandomTreeGenerator import BottomUpBinaryTreeGenerator
from src.EventLog import EventLog
from src.Population import Population

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
        
        Parameters:
            parent1 (ProcessTree): The first parent process tree.
            parent2 (ProcessTree): The second parent process tree.
            max_attempts (int, optional): Maximum number of attempts to find a valid crossover point. Defaults to 5.

        Returns:
            ProcessTree: A new process tree produced by the crossover operation, or a randomly
                     selected parent if no valid crossover was achieved.
        """
        # Create of copy of the parent trees
        parent1_copy = deep_copy_tree(parent1)
        parent2_copy = deep_copy_tree(parent2)
        
        # Randomly select crossover points
        subtree1 = random.choice(parent1_copy.get_all_operator_nodes())
        subtree2 = random.choice(parent2_copy.get_all_operator_nodes())
        
        # Try both orders of crossover replacements
        for candidate, swap_from, swap_to in [(parent1_copy, subtree1, subtree2), (parent2_copy, subtree2, subtree1)]:
            # Swap the subtree at the crossover point
            for node in candidate.get_all_operator_nodes():
                if swap_from in node.children:
                    idx = node.children.index(swap_from)
                    node.children[idx] = swap_to
                    node.children[idx].parent = node
                
            try:
                # Ensure that the tree is strictly valid
                candidate.remove_duplicate_activities()
                candidate.if_missing_insert_activities(self.EventLog.unique_activities())
                
                return candidate
            except ValueError:
                continue   # remove_duplicate_activities raise an error, i.e. not possible to remove duplicate activities without breaking the tree
        
        # If no valid crossover point was found, return a random parent
        return random.choice([parent1, parent2])

    def mutation(self, process_tree: ProcessTree) -> ProcessTree:
        """
        Performs randomly one of the following mutations on a process tree:
        - Operator swap: Selects a random operator node and changes its operator type.
        - Subtree removal: Selects a random subtree and replaces it with a new random tree
            at a random point in the tree.
        - Node addition: Selects a random operator node and adds a new leaf node to it.

        Args:
            process_tree (ProcessTree)

        Returns:
            ProcessTree: The mutated process tree.
        """
        
        def operator_swap(tree: ProcessTree) -> ProcessTree:
            # Select a random node to perform swap 
            nodes = tree.get_all_operator_nodes()
            node = random.choice(nodes)
            
            # Sample random operator and insert it to ensure valid tree
            if len(node.children) == 1:
                operators = [Operator.SEQUENCE, Operator.XOR]
                operators.remove(node.operator)
                node.operator = operators[0]
            elif len(node.children) >= 2:
                operators = [Operator.SEQUENCE, Operator.XOR, Operator.PARALLEL, Operator.LOOP]
                operators.remove(node.operator)
                node.operator = random.choice(operators)
                random.shuffle(node.children)            
            
            return tree
                  
        def subtree_removal(tree: ProcessTree) -> ProcessTree:
            # Select a random operator node to remove
            node = random.choice(tree.get_all_operator_nodes())
            
            # Ensure that is not the root node
            if not node.parent:
                return tree
            
            # Attempt to remove subtree and obtain valid tree
            node.parent.children.remove(node)
            if not node.parent.is_valid():
                node.parent.children.append(node)
                return tree
            
            # After succesfully removing subtree, generate random tree containing all missing activities
            generator = BottomUpBinaryTreeGenerator()
            missing_activities = tree.get_missing_activities(self.EventLog.unique_activities())
            new_sub_tree = generator.generate_population(missing_activities, n=1)[0]
            
            # Insert the new subtree into the tree
            insertion_node = random.choice(tree.get_all_operator_nodes())            
            insertion_node.add_child(new_sub_tree)
            
            return tree
        
        def leaf_addition(tree: ProcessTree) -> ProcessTree:            
            leaf = random.choice(tree.get_all_leaf_nodes())
            leaf.parent.children.remove(leaf)
            
            if not leaf.parent.is_valid():
                leaf.parent.children.append(leaf)
                return tree
                
            # Find a random operator node to add the leaf to
            operator = random.choice(tree.get_all_operator_nodes())
            operator.add_child(leaf)

            return tree

        def loop_addition(tree: ProcessTree) -> ProcessTree:
            # Select a loop node and a tau node
            loops = [node for node in tree.get_all_operator_nodes() if node.operator == Operator.LOOP]
            
            # Check that loops exists
            if len(loops) == 0:
                return tree
            
            # if there already exist a tau node in the loop, then skip
            loop = random.choice(loops)
            for child in loop.children:
                if child.label is not None and "tau" in child.label:
                    return tree
            
            loop.add_tau_leaf()
            
            return tree
        
        pt_copy = deep_copy_tree(process_tree)
        mutation_type = random.choice(['subtree_removal', 'leaf_addition', 'operator_swap'])

        if mutation_type == 'operator_swap':
            new_tree = operator_swap(pt_copy)
        elif mutation_type == 'subtree_removal':
            new_tree = subtree_removal(pt_copy)
        elif mutation_type == 'leaf_addition':
            new_tree = leaf_addition(pt_copy)
            
        return new_tree
    
    def generate_new_population(self, old_population: Population) -> Population:
        new_population = Population([])
        population_size = len(old_population.get_population())
        
        # Add elite trees
        elite = old_population.get_elite(int(population_size * self.elite_rate))
        new_population.add_trees(elite)

        # Add random trees
        random_count = int(population_size * self.random_creation_rate)
        new_population.add_trees(self.random_creation(random_count))
        
        # Add crossover trees
        crossover_count = int(population_size * self.crossover_rate)
        for _ in range(crossover_count):
            parent1, parent2 = random.sample(old_population.get_population(), k=2)
            new_population.add_tree(self.crossover(parent1, parent2))
                
        # Add mutation trees
        mutation_count = int(population_size * self.mutation_rate)
        for _ in range(mutation_count):
            parent = random.choice(old_population.get_population())
            new_population.add_tree(self.mutation(parent))
                    
        return new_population
    
    
def deep_copy_tree(node: ProcessTree) -> 'ProcessTree':
    # Create root node
    new_node = ProcessTree(operator=node.operator, label=node.label)
    
    # Recursively copy each child and update the parent pointer.
    for child in node.children:
        new_child = deep_copy_tree(child)
        new_child.parent = new_node
        new_node.children.append(new_child)
    return new_node