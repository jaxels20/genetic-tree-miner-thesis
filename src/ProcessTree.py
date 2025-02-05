from enum import Enum
from typing import List, Optional, Tuple
import os
import pm4py
from pm4py.objects.process_tree.obj import ProcessTree as PM4PyProcessTree, Operator as PM4PyOperator
from pm4py.objects.process_tree.exporter.variants import ptml as PM4PyExporter
from pm4py.objects.process_tree.importer.variants import ptml as PM4PyImporter

import pm4py.visualization.process_tree.visualizer as vis_process_tree
import pm4py.objects.conversion.process_tree.converter as tree_converter

class Operator(Enum):
    SEQUENCE = 'SEQ'
    XOR = 'XOR'
    PARALLEL = 'AND'
    LOOP = 'O'
    OR = 'OR'
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
        
        map = {
            Operator.SEQUENCE: PM4PyOperator.SEQUENCE,
            Operator.XOR: PM4PyOperator.XOR,
            Operator.PARALLEL: PM4PyOperator.PARALLEL,
            Operator.LOOP: PM4PyOperator.LOOP,
            Operator.OR: PM4PyOperator.OR,
            Operator.INTERLEAVING: PM4PyOperator.INTERLEAVING
        }
        
        if self.operator is None:
            return None
        return map[self.operator]

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
        
        map = {
            PM4PyOperator.SEQUENCE: Operator.SEQUENCE,
            PM4PyOperator.XOR: Operator.XOR,
            PM4PyOperator.PARALLEL: Operator.PARALLEL,
            PM4PyOperator.LOOP: Operator.LOOP,
            PM4PyOperator.OR: Operator.OR,
            PM4PyOperator.INTERLEAVING: Operator.INTERLEAVING
        }
        
        if pm4py_operator is None:
            return None
        return map[pm4py_operator]

    def __str__(self):
        children_str = ",".join(str(child) for child in self.children) if self.children else ""
        
        if self.operator is None:
            return self.label
        
        return f"{self.operator}({children_str})"
    
    @staticmethod
    def load(filename: str):
        pm4py_tree = PM4PyImporter.apply(filename)
        return ProcessTree.from_pm4py(pm4py_tree)
    
    def save(self, filename: str, format: str = "ptml"):
        pm4py_tree = self.to_pm4py()
        if format == "ptml":
            PM4PyExporter.apply(pm4py_tree, filename)
        elif format == "png":
            gviz = vis_process_tree.apply(pm4py_tree)
            vis_process_tree.save(gviz, filename + ".png")
        else:
            raise ValueError("Invalid format")
    
    def visualize(self):
        pm4py_tree = self.to_pm4py()
        gviz = vis_process_tree.apply(pm4py_tree)
        vis_process_tree.view(gviz) 
    
    def is_valid(self):
        """
        Checks if a process tree is valid according to the definition of a process tree.

        Returns:
            bool: True if the tree is valid, False otherwise.
        """
        def validate_node(node):
            # Check that activity nodes have no children
            if node.operator is None:  # Leaf node
                return len(node.children) == 0  # Leaf nodes should not have children
            
            # Check that internal nodes are operators
            if not isinstance(node.operator, Operator):
                return False  # Internal nodes must have a valid operator
            
            child_count = len(node.children)
            
            # Validate based on operator type
            if node.operator in [Operator.SEQUENCE, Operator.XOR, Operator.PARALLEL] and child_count < 1:
                return False   # SEQUENCE, XOR, and PARALLEL nodes must have at least 1 child to be a valid tree
            elif node.operator in [Operator.OR, Operator.LOOP] and child_count < 2:
                return False   # OR and LOOP nodes must have at least 2 children to be a valid tree
            
            return all(validate_node(child) for child in node.children)

        return validate_node(self)

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

    def get_all_nodes(self) -> List['ProcessTree']:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes
    
if __name__ == "__main__":
    # Example usage
    tree = ProcessTree(operator=Operator.SEQUENCE, label=None)
    tree.add_child(ProcessTree(operator=Operator.XOR, label=None))
    tree.add_child(ProcessTree(operator=Operator.PARALLEL, label=None))
    tree.children[0].add_child(ProcessTree(operator=None, label="A"))
    tree.children[0].add_child(ProcessTree(operator=None, label="B"))
    tree.children[1].add_child(ProcessTree(operator=None, label="C"))
    
    tree.visualize()
    print(tree.is_valid())
    tree.save("test.ptml")
    new_tree = ProcessTree.load("test.ptml")
    new_tree.visualize()
    print(new_tree.is_valid())