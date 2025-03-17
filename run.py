from src.GeneticAlgorithm import GeneticAlgorithm
from src.Mutator import Mutator
from src.EventLog import EventLog
import time
from src.Evaluator import SingleEvaluator
from src.PetriNet import PetriNet
from src.Filtering import Filtering


if __name__ == "__main__":
    eventlog = EventLog().from_trace_list(["ABC", "ABD", "ABD", "ABC"])
    filtered_log = Filtering.filter_eventlog_by_top_percentage_unique(eventlog, 0.5, include_all_activities=True)
    
    random_log = Filtering.filter_eventlog_random(eventlog, 0.5, include_all_activities=True)
    print(random_log)