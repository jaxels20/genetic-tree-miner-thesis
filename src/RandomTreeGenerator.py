import random
from src.EventLog import EventLog
from typing import List, Dict, Tuple, Set
from src.ProcessTree import ProcessTree, Operator
from src.Population import Population
from src.Filtering import Filtering
import pm4py


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
        
class SequentialTreeGenerator:
    def __init__(self):
        self.footprint_matrix = None
        self.unique_activities = None 

    def generate_sequential_model(self) -> ProcessTree:
        root = ProcessTree(operator=Operator.SEQUENCE)
        available_activities = list(self.unique_activities)
        random.shuffle(available_activities)

        pairs = []
        while len(available_activities) > 1:
            a = available_activities.pop()
            b = available_activities.pop()
            pairs.append((a, b))

        if available_activities:  # Handle odd number of activities
            leftover_activity = available_activities.pop()
            random_insertion_point = random.randint(0, len(pairs))
            pairs.insert(random_insertion_point, (leftover_activity,))

        for pair in pairs:
            if len(pair) == 1:
                root.add_child(ProcessTree(label=pair[0]))
            else:
                a, b = pair
                forward_relation = self.footprint_matrix.get((a, b))
                backward_relation = self.footprint_matrix.get((b, a))
                
                if forward_relation == '>':
                    operator_node = ProcessTree(operator=Operator.SEQUENCE)
                    operator_node.add_child(ProcessTree(label=a))
                    operator_node.add_child(ProcessTree(label=b))
                elif forward_relation == '<':
                    operator_node = ProcessTree(operator=Operator.SEQUENCE)
                    operator_node.add_child(ProcessTree(label=b))
                    operator_node.add_child(ProcessTree(label=a))
                elif forward_relation == '||':
                    operator_node = ProcessTree(operator=Operator.PARALLEL)
                    operator_node.add_child(ProcessTree(label=a))
                    operator_node.add_child(ProcessTree(label=b))
                else:
                    operator_node = ProcessTree(operator=Operator.XOR)
                    operator_node.add_child(ProcessTree(label=a))
                    operator_node.add_child(ProcessTree(label=b))
                
                root.add_child(operator_node)

        return root

    def generate_population(self, eventlog: EventLog, n: int) -> List[ProcessTree]:
        trees = []
        self.footprint_matrix = eventlog.get_footprint_matrix()             
        self.unique_activities = eventlog.unique_activities()
        for _ in range(n):
            tree = self.generate_sequential_model()
            trees.append(tree)

        population = Population(trees)
        return population

class InjectionTreeGenerator:
    def __init__(self, log_filtering: float):
        self.log_filtering = log_filtering

    def generate_injection_model(self, eventlog: EventLog) -> ProcessTree:
        filtered_log = Filtering.filter_eventlog_random(eventlog, self.log_filtering, include_all_activities=True)
        pm4py_log = filtered_log.to_pm4py()

        # use pm4py to generate the process tree by inductive miner on the pm4py log
        process_tree = pm4py.discover_process_tree_inductive(pm4py_log)
        # convert pm4py process tree to our ProcessTree object
        process_tree = ProcessTree.from_pm4py(process_tree)

        return process_tree

    def generate_population(self, event_log: EventLog, n: int) -> List[ProcessTree]:
        """
        Generates a population of process trees by injecting noise into the event log.
        """
        trees = []
        for _ in range(n):
            tree = self.generate_injection_model(event_log)
            if not any (trees[i].is_equal(tree) for i in range(len(trees))):
                # Ensure uniqueness of trees
                trees.append(tree)
        
        if len(trees) < n:
            generator = BottomUpBinaryTreeGenerator()
            unique_activities = event_log.unique_activities()

        while(len(trees) < n):
            tree = generator._generate_naive_binary_tree(unique_activities)
            if not any (trees[i].is_equal(tree) for i in range(len(trees))):
                # Ensure uniqueness of trees
                trees.append(tree)

        population = Population(trees)
        return population
    
if __name__ == "__main__":
    unique_activities = ["A", "B", "C"]
    generator = BottomUpBinaryTreeGenerator()
    tree = generator.generate_population(unique_activities, 100)[0]
    print(tree)
