from src.EventLog import EventLog
from src.Mutator import Mutator, TournamentMutator
from src.GeneticAlgorithm import GeneticAlgorithm
from pm4py.algo.discovery.inductive.algorithm import apply as pm4py_inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from src.PetriNet import PetriNet
import numpy as np

class Discovery:
    @staticmethod
    def genetic_algorithm(event_log: EventLog, **kwargs) -> PetriNet:
        """
        A wrapper for the genetic algorithm.
        """
        ga = GeneticAlgorithm(
            method_name=kwargs.get("method_name"),
        )
        
        # Args for the run algorithm
        population_size=kwargs.get("population_size")
        max_generations = kwargs.get("max_generations", 1_000_000_000)
        min_fitness=kwargs.get("min_fitness", None)
        stagnation_limit=kwargs.get("stagnation_limit", None)
        time_limit=kwargs.get("time_limit", None)
        generator = kwargs.get("generator")
        percentage_of_log = kwargs.get("percentage_of_log")
        mutator = kwargs.get("mutator")
        objective = kwargs.get("objective")
        export_monitor_path = kwargs.get("export_monitor_path", None)
        
        our_pt = ga.run(
            eventlog=event_log, 
            population_size=population_size,
            mutator=mutator,
            generator=generator, 
            objective=objective, 
            percentage_of_log=percentage_of_log, 
            max_generations=max_generations, 
            min_fitness=min_fitness,
            stagnation_limit=stagnation_limit,
            time_limit=time_limit,
            export_monitor_path=export_monitor_path,
        )
        pm4py_net, init, end = our_pt.to_pm4py_pn()
        
        if kwargs.get("return_monitor", False):
            # Return the monitor as well
            return PetriNet.from_pm4py(pm4py_net, init, end), ga.monitor
        
        return PetriNet.from_pm4py(pm4py_net, init, end)

    @staticmethod
    def inductive_miner(event_log: EventLog, **kwargs) -> PetriNet:
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
        
        return method(event_log, **kwargs)
    
    
    
    