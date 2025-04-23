import random
from typing import List
from src.ProcessTree import ProcessTree, Operator
from src.RandomTreeGenerator import BottomUpRandomBinaryGenerator
from src.EventLog import EventLog
from src.Population import Population

class MutatorBase:
    def __init__(self):
        pass

    def generate_new_population(self, old_population: List[ProcessTree], new_population_size: int) -> List[ProcessTree]:
        raise NotImplementedError


class Mutator(MutatorBase):
    def __init__(self, random_creation_rate: float, crossover_rate: float, mutation_rate: float, elite_rate: float):
        self.event_log = None
        self.random_creation_rate = random_creation_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        self.mutation_types_weights = None
        
    def set_event_log(self, event_log: EventLog):
        self.event_log = event_log
        
    def random_creation(self, num_new_trees: int) -> List[ProcessTree]:
        generator = BottomUpRandomBinaryGenerator()
        new_trees = generator.generate_population(self.event_log.unique_activities(), num_new_trees)
        return new_trees.get_population()
        
    def crossover(self, parent1: ProcessTree, parent2: ProcessTree) -> ProcessTree:
        """
        Performs subtree crossover between two process trees.
        
        Parameters:
            parent1 (ProcessTree): The first parent process tree.
            parent2 (ProcessTree): The second parent process tree.

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
                candidate.if_missing_insert_activities(self.event_log.unique_activities())
            except ValueError:
                continue   # remove_duplicate_activities raise an error, i.e. not possible to remove duplicate activities without breaking the tree
            
            if candidate.is_valid():
                return candidate
        
        # If no valid crossover point was found, return a random parent
        return deep_copy_tree(random.choice([parent1, parent2]))

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
            swap_node = random.choice(nodes)
            
            non_tau_children, taus = [], []
            for child in swap_node.children:
                (taus if not child.label and not child.operator else non_tau_children).append(child)
            
            # Do operator swap according to the operator type
            curr_operator = swap_node.operator
 
            if len(non_tau_children) >= 2:
                if curr_operator == Operator.LOOP:
                    swap_node.operator = random.choice([Operator.SEQUENCE, Operator.XOR, Operator.PARALLEL])
                    for tau in taus:
                        swap_node.children.remove(tau)
                else:
                    swap_node.operator = random.choice([Operator.SEQUENCE, Operator.XOR, Operator.PARALLEL, Operator.LOOP])
                    
            elif len(non_tau_children) == 1:
                if curr_operator == Operator.LOOP:
                    swap_node.operator = random.choice([Operator.SEQUENCE, Operator.XOR])
                    for tau in taus:
                        swap_node.children.remove(tau)
                else:
                    if curr_operator == Operator.SEQUENCE:
                        swap_node.operator = Operator.XOR
                    else:
                        swap_node.operator = Operator.SEQUENCE
                                                        
            random.shuffle(swap_node.children)        
            
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
            generator = BottomUpRandomBinaryGenerator()
            missing_activities = tree.get_missing_activities(self.event_log.unique_activities())
            new_sub_tree = generator.generate_population(missing_activities, n=1)[0]
            
            # Insert the new subtree into the tree
            insertion_node = random.choice(tree.get_all_operator_nodes())            
            insertion_node.add_child(new_sub_tree)
            
            return tree
        
        def leaf_addition(tree: ProcessTree) -> ProcessTree:
            activities = [node for node in tree.get_all_leaf_nodes() if node.label is not None]
            leaf = random.choice(activities)
            
            # Attempt to remove leaf without breaking structure
            leaf.parent.children.remove(leaf)
            if not leaf.parent.is_valid():
                leaf.parent.children.append(leaf)
                return tree
                
            # Find a random operator node to add the leaf to
            operator = random.choice(tree.get_all_operator_nodes())
            operator.add_child(leaf)

            return tree

        def loop_addition(tree: ProcessTree) -> ProcessTree:
            # Select a random leaf node
            leaves = [node for node in tree.get_all_leaf_nodes() if node.label is not None]
            new_loop = random.choice(leaves)
            
            # Replace the leaf node with a loop node
            activity = new_loop.label
            new_loop.label = None
            new_loop.operator = Operator.LOOP
            
            # Add tau node to new_loop
            tau = ProcessTree(parent=new_loop, operator=None, label=None)
            new_loop.add_child(tau)
            new_loop.add_child(ProcessTree(parent=new_loop, label=activity))
            random.shuffle(new_loop.children)
            
            return tree
        
        pt_copy = deep_copy_tree(process_tree)
        mutation_type = random.choice(['loop_addition', 'operator_swap', 'subtree_removal', 'leaf_addition'])
        
        # define discrete distribution for mutation types
        # mutation_type = random.choices(
        #     ['loop_addition', 'operator_swap', 'subtree_removal', 'leaf_addition'],
        #     weights=[0.25, 0.25, 0.25, 0.25]
        # )[0]

        if mutation_type == 'operator_swap':
            new_tree = operator_swap(pt_copy)
        elif mutation_type == 'subtree_removal':
            new_tree = subtree_removal(pt_copy)
        elif mutation_type == 'leaf_addition':
            new_tree = leaf_addition(pt_copy)
        elif mutation_type == 'loop_addition':
            new_tree = loop_addition(pt_copy)
            
        return new_tree
    
    def generate_new_population(self, old_population: Population) -> Population:
        new_population = Population([])
        population_size = len(old_population.get_population())

        # Add elite trees
        elite = old_population.get_best_trees(int(population_size * self.elite_rate))
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
            
        # Ensure the new population size is the same as the old one
        if len(new_population) < population_size:
            missing_trees = population_size - len(new_population)
            random_trees = self.random_creation(missing_trees)
            new_population.add_trees(random_trees)
            
        return new_population

class TournamentMutator(Mutator):
    def __init__(self, random_creation_rate: float, crossover_rate: float, mutation_rate: float, elite_rate: float, tournament_size: float):
        super().__init__(random_creation_rate, crossover_rate, mutation_rate, elite_rate)
        self.tournament_size = tournament_size

    def generate_new_population(self, old_population: Population) -> Population:
        new_population = Population([])
        population_size = len(old_population.get_population())
        
        # Elite selection
        elite_population = old_population.get_best_trees(int(self.elite_rate * population_size))
        new_population.add_trees(elite_population)

        # Tournament selection
        tournament_population = old_population.get_population_interval(0, self.tournament_size)
        for _ in range(int(self.crossover_rate * population_size)):
            random_sample = random.sample(tournament_population, k=6)
            tree1, tree2 = sorted(random_sample, key=lambda tree: tree.get_fitness(), reverse=True)[:2]
            new_population.add_tree(self.crossover(tree1, tree2))
        
        # Mutation selection
        mutation_population = old_population.get_population_interval(self.elite_rate, 1)
        for _ in range(int(self.mutation_rate * population_size)):
            tree = random.choice(mutation_population)
            new_population.add_tree(self.mutation(tree))
            
        # Random creation
        random_population = self.random_creation(int(self.random_creation_rate * population_size))
        new_population.add_trees(random_population)
        
        # Ensure the new population size is the same as the old one
        if len(new_population) < population_size:
            missing_trees = population_size - len(new_population)
            random_trees = self.random_creation(missing_trees)
            new_population.add_trees(random_trees)
        
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