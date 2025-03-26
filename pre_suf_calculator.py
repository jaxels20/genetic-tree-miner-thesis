import pm4py
import pandas as pd
from itertools import combinations

# Read the .xes file using pm4py
event_log = pm4py.read_xes("real_life_datasets/Sepsis/Sepsis.xes")

# Extract traces based on the case IDs (assuming 'case:concept:name' is the column for case IDs)
traces = []
for case_id, case_events in event_log.groupby('case:concept:name'):
    # Extract the event names (activity names) from the case events
    trace_activities = case_events['concept:name'].tolist()
    traces.append(trace_activities)


def longest_common_prefix(trace1, trace2):
    """Returns the longest common prefix length between two traces."""
    common = 0
    for e1, e2 in zip(trace1, trace2):
        if e1 == e2:
            common += 1
        else:
            break
    return common

def longest_common_suffix(trace1, trace2):
    """Returns the longest common suffix length between two traces."""
    return longest_common_prefix(trace1[::-1], trace2[::-1])

def compute_similarity(log, method="prefix"):
    """Computes average prefix or suffix similarity across all trace pairs."""
    total_similarity = 0
    count = 0

    for trace1, trace2 in combinations(log, 2):
        if method == "prefix":
            common_length = longest_common_prefix(trace1, trace2)
        else:
            common_length = longest_common_suffix(trace1, trace2)

        min_length = min(len(trace1), len(trace2))
        similarity = common_length / min_length  # Normalize similarity
        total_similarity += similarity
        count += 1

    return total_similarity / count if count > 0 else 0

# Compute prefix and suffix similarity
prefix_similarity = compute_similarity(traces, method="prefix")
suffix_similarity = compute_similarity(traces, method="suffix")

# Determine best evaluation method
best_method = "Prefix-based token replay" if prefix_similarity > suffix_similarity else "Suffix-based token replay"

# Print results
print(f"Average Prefix Similarity: {prefix_similarity:.4f}")
print(f"Average Suffix Similarity: {suffix_similarity:.4f}")
print(f"Recommended Evaluation Method: {best_method}")
