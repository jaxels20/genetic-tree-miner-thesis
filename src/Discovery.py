from src.EventLog import EventLog
from src.Mutator import Mutator, TournamentMutator
from src.GeneticAlgorithm import GeneticAlgorithm
from pm4py.algo.discovery.inductive.algorithm import apply as pm4py_inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from src.PetriNet import PetriNet
from src.ProcessTree import ProcessTree
import numpy as np

class Discovery:
    @staticmethod
    def genetic_algorithm(event_log: EventLog, **kwargs) -> tuple[PetriNet, ProcessTree]:
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
        percentage_of_log = kwargs.get("percentage_of_log", None)
        mutator = kwargs.get("mutator")
        objective = kwargs.get("objective")
        export_monitor_path = kwargs.get("export_monitor_path", None)
        export_decomposed_objective_function_path = kwargs.get("export_decomposed_objective_function_path", None)
        
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
            export_decomposed_objective_function_path=export_decomposed_objective_function_path
        )
        pm4py_net, init, end = our_pt.to_pm4py_pn()
        
        return PetriNet.from_pm4py(pm4py_net, init, end), our_pt

    @staticmethod
    def inductive_miner(event_log: EventLog, **kwargs) -> PetriNet:
        """
        A wrapper for the inductive miner.
        """
        event_log = event_log.to_pm4py()
        pm4py_pt = pm4py_inductive_miner(event_log)
        pm4py_net, init, end = pt_converter.apply(pm4py_pt, variant=pt_converter.Variants.TO_PETRI_NET)
        
        return PetriNet.from_pm4py(pm4py_net, init, end)
