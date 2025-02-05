from enum import Enum
from typing import List, Optional, Tuple
import pm4py
from pm4py.objects.process_tree.obj import ProcessTree as PM4PyProcessTree, Operator as PM4PyOperator

import pm4py.visualization.process_tree.visualizer as vis_process_tree
import pm4py.objects.conversion.process_tree.converter as tree_converter

class Operator(Enum):
    SEQUENCE = '->'
    XOR = 'X'
    PARALLEL = '+'
    LOOP = '*'
    OR = 'O'
    INTERLEAVING = "<>"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

class ProcessTree:
    def __init__(self, operator: Optional[Operator] = None, label: Optional[str] = None, parent: Optional['ProcessTree'] = None, children: Optional[List['ProcessTree']] = None):
        self.operator = operator
        self.label = label
        self.parent = parent
        self.children = children if children is not None else []
        
        # Used for Genetic Algorithm
        self.fitness = None

    def add_child(self, child: 'ProcessTree'):
        child.parent = self
        self.children.append(child)

    def to_pm4py(self) -> PM4PyProcessTree:
        pm4py_node = PM4PyProcessTree(operator=self._to_pm4py_operator(), label=self.label)
        for child in self.children:
            pm4py_child = child.to_pm4py()
            pm4py_child.parent = pm4py_node
            pm4py_node.children.append(pm4py_child)
        return pm4py_node

    def _to_pm4py_operator(self):
        if self.operator is None:
            return None
        return getattr(PM4PyOperator, self.operator.name, None)

    @staticmethod
    def from_pm4py(pm4py_tree: PM4PyProcessTree) -> 'ProcessTree':
        root = ProcessTree(operator=ProcessTree._from_pm4py_operator(pm4py_tree.operator), label=pm4py_tree.label)
        for pm4py_child in pm4py_tree.children:
            child = ProcessTree.from_pm4py(pm4py_child)
            child.parent = root
            root.children.append(child)
        return root

    @staticmethod
    def _from_pm4py_operator(pm4py_operator):
        if pm4py_operator is None:
            return None
        for op in Operator:
            if op.value == pm4py_operator.value:
                return op
        return None

    def __str__(self):
        children_str = ",".join(str(child) for child in self.children) if self.children else ""
        
        if self.operator is None:
            return self.label
        
        return f"{self.operator}({children_str})"
    
    def load(self, filename: str):
        raise NotImplementedError
    
    def save(self, filename: str, format: str = None):
        pm4py_tree = self.to_pm4py()
        
        raise NotImplementedError
    
    def visualize(self):
        pm4py_tree = self.to_pm4py()
        gviz = vis_process_tree.apply(pm4py_tree)
        vis_process_tree.view(gviz) 
    
    # TODO: Create a function to check if the tree is valid
    def is_valid(self):
        raise NotImplementedError

    def to_pm4py_pn(self): #-> Tuple[pm4py.objects.petri, pm4py.objects.petri.common.final_marking, pm4py.objects.petri.common.initial_marking]:
        pm4py_tree = self.to_pm4py()
        return tree_converter.apply(pm4py_tree)

    def set_fitness(self, fitness):
        self.fitness = fitness
    
    def get_fitness(self):
        return self.fitness

    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __le__(self, other):
        return self.fitness <= other.fitness
    
    def __gt__(self, other):
        return self.fitness > other.fitness
    
    def __ge__(self, other):
        return self.fitness >= other.fitness

if __name__ == "__main__":
    # Example usage
    tree = ProcessTree(operator=Operator.SEQUENCE, label=None)
    tree.add_child(ProcessTree(operator=Operator.XOR, label=None))
    tree.add_child(ProcessTree(operator=Operator.PARALLEL, label=None))
    tree.children[0].add_child(ProcessTree(operator=None, label="A"))
    tree.children[0].add_child(ProcessTree(operator=None, label="B"))
    tree.children[1].add_child(ProcessTree(operator=None, label="C"))
    
    pm4py_tree = tree.to_pm4py()
    print(pm4py_tree)
    gviz = vis_process_tree.apply(pm4py_tree)
    #vis_process_tree.view(gviz)
    

    tree = ProcessTree.from_pm4py(pm4py_tree)
    print(tree)