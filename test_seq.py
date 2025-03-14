from src.RandomTreeGenerator import SequentialTreeGenerator
from src.EventLog import EventLog

if __name__ == "__main__":
    generator = SequentialTreeGenerator()
    eventlog = EventLog.from_trace_list(["ABC", "ACB"])
    tree = generator.generate_trace_model(eventlog)
    print(tree)