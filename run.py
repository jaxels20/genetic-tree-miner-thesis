from src.GeneticAlgorithm import GeneticAlgorithm
from src.Mutator import Mutator
from src.EventLog import EventLog
import time





if __name__ == "__main__":
    
    # import cProfile
    # import pstats
    
    eventlog = EventLog.from_trace_list(["ABBBC"])
    mutator = Mutator(eventlog, random_creation_rate=0.1, crossover_rate=0.2, mutation_rate=0.5, elite_rate=0.2)
    ga = GeneticAlgorithm(mutator, min_fitness=None, max_generations=200, stagnation_limit=None, time_limit=90, population_size=100)
    start = time.time()
    best_tree = ga.run(eventlog=eventlog)
    
    
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.strip_dirs().sort_stats("cumulative").print_stats(10)
    
    # Print results
    print(f"Time taken: {time.time() - start}")
    print(f"Best tree: {best_tree}")
    print(f"Best tree fitness: {best_tree.get_fitness()}")
    best_tree.visualize()
    pm4py_pn, init, end = best_tree.to_pm4py_pn()