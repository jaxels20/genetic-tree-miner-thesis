from src.EventLog import EventLog
import random
class Filtering:
    def __init__():
        pass
    
    @staticmethod
    def filter_eventlog_by_top_percentage_unique(eventlog: EventLog, percentage: float) -> EventLog:
        """
        Filters an event log to include only the top percentage of most frequent traces.
        
        Parameters:
        -----------
        eventlog : EventLog
            The input event log to filter.
        percentage : float
            The percentage (as a fraction between 0 and 1 or as a percentage > 1) of most frequent unique traces to keep.
            For example, use 0.2 or 20 for the top 20% of unique traces.
        
        Returns:
        --------
        EventLog
            A new EventLog containing only the traces that are among the top percentage by frequency.
        """
        # If percentage is given as a value greater than 1, convert it (e.g., 20 becomes 0.2)
        if percentage > 1:
            percentage = percentage / 100.0

        # Dictionary to count frequency of each trace signature
        # Here, the signature is defined as the tuple of event activities.
        signature_freq = {}
        for trace in eventlog.traces:
            signature = tuple(event.activity for event in trace.events)
            signature_freq[signature] = signature_freq.get(signature, 0) + 1

        # Sort the signatures by frequency in descending order.
        sorted_signatures = sorted(signature_freq.items(), key=lambda x: x[1], reverse=True)

        # Determine how many unique signatures to keep.
        total_unique = len(sorted_signatures)
        num_to_keep = max(1, int(round(percentage * total_unique)))

        # Extract the set of top signatures.
        top_signatures = {signature for signature, _ in sorted_signatures[:num_to_keep]}

        # Build the filtered event log by including traces with a signature in top_signatures.
        filtered_log = EventLog()
        for trace in eventlog.traces:
            signature = tuple(event.activity for event in trace.events)
            if signature in top_signatures:
                filtered_log.traces.append(trace)
                
        return filtered_log
    
    @staticmethod
    def filter_eventlog_by_top_percentage(eventlog: EventLog, percentage: float) -> EventLog:
        """
        Filters an event log to include only the top percentage of most frequent traces (by total occurrences).

        Parameters:
        -----------
        eventlog : EventLog
            The input event log to filter.
        percentage : float
            The percentage (as a fraction between 0 and 1 or as a percentage > 1) of the total traces to keep.

        Returns:
        --------
        EventLog
            A new EventLog containing traces that make up the top percentage of occurrences.
        """
        if percentage > 1:
            percentage = percentage / 100.0

        # Count frequency of each unique trace signature
        signature_freq = {}
        for trace in eventlog.traces:
            signature = tuple(event.activity for event in trace.events)
            signature_freq[signature] = signature_freq.get(signature, 0) + 1

        # Sort traces by total frequency (descending)
        sorted_signatures = sorted(signature_freq.items(), key=lambda x: x[1], reverse=True)

        # Compute total number of traces in the original event log
        total_traces = len(eventlog.traces)
        num_to_keep = int(round(percentage * total_traces))

        # Select traces until we reach the required total count
        filtered_log = EventLog()
        kept_traces = 0
        for signature, freq in sorted_signatures:
            for trace in eventlog.traces:
                if kept_traces >= num_to_keep:
                    return filtered_log  # Stop early if we reach 50%
                if tuple(event.activity for event in trace.events) == signature:
                    filtered_log.traces.append(trace)
                    kept_traces += 1

        return filtered_log

    @staticmethod
    def filter_eventlog_random(eventlog: EventLog, percentage: float) -> EventLog:
        """
        Filters an event log to include a random subset of traces.
        
        Parameters:
        -----------
        eventlog : EventLog
            The input event log.
        percentage : float
            The percentage of traces to keep. It can be a fraction (e.g., 0.2 for 20%)
            or a percentage greater than 1 (e.g., 20 for 20%).
        
        Returns:
        --------
        EventLog
            A new EventLog containing a random subset of traces.
        """
        # Convert percentage if provided as a value greater than 1.
        if percentage > 1:
            percentage = percentage / 100.0

        total_traces = len(eventlog.traces)
        # Ensure at least one trace is kept.
        num_to_keep = max(1, int(round(percentage * total_traces)))
        
        # Randomly sample the desired number of traces.
        selected_traces = random.sample(eventlog.traces, num_to_keep)
        
        # Build the filtered event log.
        filtered_log = EventLog()
        filtered_log.traces = selected_traces
        
        return filtered_log
    

        """
        Filters an event log to include all traces whose signature is among the top percentage of unique trace signatures.
        In other words, it finds the top X% most frequent unique trace signatures and returns all traces with any of those signatures.
        
        Parameters:
        -----------
        eventlog : EventLog
            The input event log to filter.
        percentage : float
            The percentage (as a fraction between 0 and 1 or as a percentage > 1) of unique trace signatures to keep.
            For example, use 0.2 or 20 for the top 20% unique trace signatures.
        
        Returns:
        --------
        EventLog
            A new EventLog containing all traces from the original log that have a top signature.
        """
        # Convert percentage if provided as a value greater than 1 (e.g., 20 becomes 0.2)
        if percentage > 1:
            percentage = percentage / 100.0

        # Count the frequency of each trace signature.
        # Here, a signature is defined as a tuple of event activity names.
        signature_freq = {}
        for trace in eventlog.traces:
            signature = tuple(event.activity for event in trace.events)
            signature_freq[signature] = signature_freq.get(signature, 0) + 1

        # Sort the unique signatures by their frequency (descending order).
        sorted_signatures = sorted(signature_freq.items(), key=lambda x: x[1], reverse=True)

        # Determine how many unique signatures to keep (ensuring at least one is kept).
        total_unique = len(sorted_signatures)
        num_to_keep = max(1, int(round(percentage * total_unique)))

        # Extract the set of top signatures.
        top_signatures = {signature for signature, _ in sorted_signatures[:num_to_keep]}

        # Build a new event log containing all traces that have a signature in the top_signatures set.
        filtered_log = EventLog()
        for trace in eventlog.traces:
            signature = tuple(event.activity for event in trace.events)
            if signature in top_signatures:
                filtered_log.traces.append(trace)

        return filtered_log