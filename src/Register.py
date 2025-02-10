from dataclasses import dataclass
from typing import Dict

@dataclass
class ProcessTreeRegister:
    """
    A data class to store process trees and their associated fitness values.
    """
    fitness_values: Dict[str, float]  # Dictionary to store fitness values (key: tree ID, value: fitness)
        
    def __getitem__(self, tree_id: str) -> float:
        """
        Retrieve the fitness value of a process tree by its ID.
        
        :param tree_id: The ID of the process tree.
        :return: The fitness value associated with the given ID.
        """
        return self.fitness_values[tree_id]
    
    def __setitem__(self, tree_id: str, fitness: float) -> None:
        """
        Set the fitness value of a process tree by its ID.
        
        :param tree_id: The ID of the process tree.
        :param fitness: The new fitness value to set.
        """
        self.fitness_values[tree_id] = fitness