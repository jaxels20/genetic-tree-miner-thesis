import pm4py
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter 
from src.EventLog import EventLog
from src.Filtering import Filtering

eventlog_path = "./real_life_datasets/2017/2017.xes"

eventlog = EventLog.load_xes(eventlog_path)
num_events_after_removal = sum([len(trace) for trace in eventlog.traces])
print(f"Number of events after removal: {num_events_after_removal}")

""" eventlog = Filtering.filter_eventlog_random(eventlog, 0.90, False)

for trace in eventlog.traces:
    for i in range(len(trace)):
        event = trace.events[i]
        event.activity = event.activity.replace(":", "_")
        trace.events[i] = event

num_events_after_removal = sum([len(trace) for trace in eventlog.traces])
print(f"Number of events after removal: {num_events_after_removal}")

eventlog.to_xes(eventlog_path.replace(".xes", "_filtered.xes"))

pm4py_log = eventlog.to_pm4py()
df = pm4py.convert_to_dataframe(pm4py_log)
df.to_csv(eventlog_path.replace(".xes", "_filtered.csv"), index=False) """
