from src.RandomTreeGenerator import SequentialTreeGenerator
from src.EventLog import EventLog
from src.RandomTreeGenerator import InjectionTreeGenerator

if __name__ == "__main__":
    generator = InjectionTreeGenerator()
    # Define 10 different traces
    trace_list = [
        "ABC", "ACB", "BAC", "BCA", "CAB", "CBA",
        "ABCD", "ADCB", "BACD", "BCDA"
    ]

    # Create the event log
    eventlog = EventLog.from_trace_list(trace_list)
    #eventlog = EventLog.from_trace_list(["ABC", "ACB"])
    tree = generator.generate_population(eventlog,5,0.1)
    #print(tree)