from src.EventLog import EventLog
from src.Mutator import Mutator
from src.GeneticAlgorithm import GeneticAlgorithm
from pm4py.algo.discovery.inductive.algorithm import apply as pm4py_inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from src.PetriNet import PetriNet


class Discovery:
    @staticmethod
    def genetic_algorithm(event_log: EventLog, random_creation_rate, crossover_rate, mutation_rate, 
                       elite_rate, min_fitness, max_generations, stagnation_limit, 
                       time_limit, population_size):
        """
        A wrapper for the genetic algorithm.
        """        
        mutator = Mutator(event_log, random_creation_rate=random_creation_rate, crossover_rate=crossover_rate,
                      mutation_rate=mutation_rate, elite_rate=elite_rate)

        ga = GeneticAlgorithm(mutator, min_fitness=min_fitness, max_generations=max_generations, 
                          stagnation_limit=stagnation_limit, time_limit=time_limit, 
                          population_size=population_size)

        best_tree = ga.run(event_log)
        pm4py_net, init, end = best_tree.to_pm4py_pn()
        
        return PetriNet.from_pm4py(pm4py_net, init, end)
    
    @staticmethod
    def inductive_miner(event_log: EventLog, **kwargs):
        """
        A wrapper for the inductive miner.
        """
        event_log = event_log.to_pm4py()
        pm4py_pt = pm4py_inductive_miner(event_log)
        pm4py_net, init, end = pt_converter.apply(pm4py_pt, variant=pt_converter.Variants.TO_PETRI_NET)
        
        return PetriNet.from_pm4py(pm4py_net, init, end)


    # Map method names to static methods
    methods = {
        "Genetic Miner": genetic_algorithm,
        "Inductive Miner": inductive_miner
    }


    @classmethod
    def run_discovery(cls, method_name: str, event_log: EventLog, **kwargs):
        """
        Runs the specified discovery method based on method_name.
        
        Parameters:
        - method_name (str): The name of the method to run (e.g., "alpha", "heuristic", "inductive", "aau_miner", "gnn_miner").
        - event_log (EventLog): The event log to be passed to the discovery method.
        - **kwargs: Additional arguments to be passed to the method, such as model_path for GNN_miner.
        
        Returns:
        - PetriNet: The discovered Petri net.
        """
        method = cls.methods.get(method_name)
        if method is None:
            raise ValueError(f"Discovery method '{method_name}' not found.")
        
        # Call the appropriate method, passing **kwargs for any extra arguments needed
        return method(event_log, **kwargs)
    
    
    
    