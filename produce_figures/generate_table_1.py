import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import os 
import pandas as pd

def collect_all_xes_files(root_dir):
    xes_files = []
    files = os.listdir(root_dir)
    for file in files:
        if file.endswith(".xes"):
            xes_files.append(os.path.join(root_dir, file))
                
    return xes_files 

INPUT_DIR = "./real_life_datasets"
OUTPUT_DIR = "./figures"

if __name__ == "__main__":
    files = collect_all_xes_files(INPUT_DIR)

    data = []

    for file in files:

        #log = xes_importer.apply(file)
        log = pm4py.read_xes(file)

        # Extract traces based on the case IDs (assuming 'case:concept:name' is the column for case IDs)
        traces = []
        for case_id, case_events in log.groupby('case:concept:name'):
            # Extract the event names (activity names) from the case events
            trace_activities = case_events['concept:name'].tolist()
            traces.append(trace_activities)

        traces = set()  # Use a set to store unique traces as tuples
        unique_traces_list = []  # List to store the final unique traces

        for case_id, case_events in log.groupby('case:concept:name'):
            trace_activities = tuple(case_events['concept:name'].tolist())  # Convert list to tuple (hashable)

            if trace_activities not in traces:
                traces.add(trace_activities)  # Add to the set (for uniqueness)
                unique_traces_list.append(list(trace_activities))  # Add to the final list as a list


        log = xes_importer.apply(file)
        distinct_activities = [event["concept:name"] for trace in log for event in trace]

        # Get the number of traces
        num_traces = len(log)
        # Get the number of events
        num_events = sum([len(trace) for trace in log])
        # Get the average number of events per trace
        avg_events_per_trace = num_events / num_traces
        # Get the Num of distinct activities
        num_distinct_activities = len(set(distinct_activities))
        # Get the maximum number of events in a trace
        max_events_in_trace = max([len(trace) for trace in log])
        # Get the minimum number of events in a trace
        min_events_in_trace = min([len(trace) for trace in log])
        
        # get the number of unique traces
        unique_traces = set()
        for trace in log:
            unique_traces.add(tuple([event["concept:name"] for event in trace]))
        num_unique_traces = len(unique_traces)
        
        
        # Step 1: Remove 'real_life_datasets/' and the next folder name
        file_name = os.path.split(os.path.split(file)[1])[1]  # Get the last part after the second split
        
        # Step 2: Extract the dataset name (without the file extension)
        dataset = file_name.split(".")[0]
        
        # Step 3: Apply the transformations on the dataset name
        if dataset.startswith("BPI"):
            dataset = dataset.replace("BPI", "")
        
        dataset = dataset.replace("_", "-")
        
        if dataset.startswith("-"):
            dataset = dataset[1:]
        data.append(
            {
                "Dataset": dataset,
                "Number of Traces": num_traces,
                "Average Trace Length": int(avg_events_per_trace),
                "Number of Activities": num_distinct_activities,
                "Number of Unique Traces": num_unique_traces,
            }
        )
        
    df = pd.DataFrame(data)
    # to latex
    df.to_latex(f"{OUTPUT_DIR}/table_1.tex", index=False)
