from src.GeneticAlgorithm import GeneticAlgorithm
from src.Mutator import Mutator
from src.EventLog import EventLog
import time
from src.Evaluator import SingleEvaluator
from src.PetriNet import PetriNet


if __name__ == "__main__":
    
    # import cProfile
    # import pstats
    
    eventlog = EventLog().load_xes("./controlled_scenarios/overleaf_example/eventlog.xes")
    
    mutator = Mutator(eventlog, random_creation_rate=0.1, crossover_rate=0.2, mutation_rate=0.5, elite_rate=0.2)
    ga = GeneticAlgorithm(mutator, min_fitness=None, max_generations=200, stagnation_limit=30, time_limit=90, population_size=100)
    best_tree = ga.run(eventlog=eventlog)
    pn, start, end = best_tree.to_pm4py_pn()
    
    PetriNet.from_pm4py(pn).visualize("best_tree")
    
    evaluator = SingleEvaluator(pn, start, end, eventlog)
    precision = evaluator.get_exact_matching("precision")
    print(precision)
    
    
    