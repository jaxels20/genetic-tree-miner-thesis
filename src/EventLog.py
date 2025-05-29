import lxml.etree as ET
from pm4py.objects.log.obj import EventLog as PM4PyEventLog, Trace as PM4PyTrace, Event as PM4PyEvent
import src.FastTokenBasedReplay as FastTokenBasedReplay
from datetime import datetime, timedelta

class Event:
    """
    Class representing an event in the event log.

    Attributes:
    -----------
    activity : str
        The activity name of the event.
    timestamp : str
        The timestamp of when the event occurred.
    attributes : dict
        Other event attributes (like resource, lifecycle:transition, etc.).
    """
    def __init__(self, activity: str, timestamp: str, attributes: dict):
        self.activity = activity
        self.timestamp = timestamp
        self.attributes = attributes

    def __repr__(self):
        return f"Event(activity={self.activity}, timestamp={self.timestamp}, attributes={self.attributes})"
    def __eq__(self, other):
        """Check equality based on activity and timestamp."""
        if not isinstance(other, Event):
            return False
        return (self.activity == other.activity)

class Trace:
    """
    Class representing a trace in the event log.

    Attributes:
    -----------
    trace_id : str
        The identifier of the trace.
    events : list[Event]
        A list of events that belong to this trace.
    attributes : dict
        Other trace attributes (like case ID).
    """
    def __init__(self, trace_id: str, attributes: dict):
        self.trace_id = trace_id
        self.events = []
        self.attributes = attributes

    def add_event(self, event: Event):
        """Add an event to the trace."""
        self.events.append(event)

    def __repr__(self):
        return f"Trace(trace_id={self.trace_id}, events={len(self.events)})"

    def __len__(self):
        """Return the number of events in the trace."""
        return len(self.events)

    def __eq__(self, other):
        """Check equality based events."""
        if not isinstance(other, Trace):
            return False
        return self.events == other.events
    
class EventLog:
    """
    Class representing an event log.

    Attributes:
    -----------
    traces : list[Trace]
        A list of traces in the event log.
    """
    def __init__(self):
        self.traces = []
        self._unique_activities = None
        self.name = None
    
    @staticmethod
    def load_xes(xes_file: str):
        """
        Load an event log from an XES file.

        Parameters:
        -----------
        xes_file : str
            The path to the XES file.
        """
        eventlog = EventLog()
        tree = ET.parse(xes_file)
        root = tree.getroot()

        # Iterate through the traces in the XES file
        for trace in root.findall(".//{*}trace"):
            trace_id = ""
            trace_attributes = {}

            # Parse trace attributes
            for attr in trace.findall("{*}string"):
                if attr.attrib["key"] == "concept:name":
                    trace_id = attr.attrib["value"]
                else:
                    trace_attributes[attr.attrib["key"]] = attr.attrib["value"]

            current_trace = Trace(trace_id, trace_attributes)

            # Iterate through the events in the trace
            for event in trace.findall("{*}event"):
                event_attributes = {}
                activity = ""
                timestamp = ""

                # Parse event attributes
                for attr in event:
                    if attr.attrib["key"] == "concept:name":
                        activity = attr.attrib["value"]
                    elif attr.attrib["key"] == "time:timestamp":
                        timestamp = attr.attrib["value"]
                    else:
                        event_attributes[attr.attrib["key"]] = attr.attrib["value"]

                current_event = Event(activity, timestamp, event_attributes)
                current_trace.add_event(current_event)

            # Add the trace to the log
            eventlog.traces.append(current_trace)
        return eventlog

    @staticmethod
    def from_trace_list(trace_list: list):
        """
        Create an event log from a list of traces.

        Parameters:
        -----------
        trace_list : list[Trace]
            A list of traces to add to the event log.
        """
        eventlog = EventLog()

        # Iterate through the provided trace list
        for idx, trace_str in enumerate(trace_list):
            trace_id = f"trace_{idx+1}"  # Assign a trace ID based on index
            trace = Trace(trace_id, attributes={})

            # Create an event for each activity in the trace string
            timestamp = datetime(2023, 8, 15, 10, 0, 0)
            timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            for activity in trace_str:
                event = Event(activity, timestamp=timestamp_str, attributes={})  # No timestamp or attributes
                trace.add_event(event)
                timestamp += timedelta(hours=1)   # Increment timestamp by 1 hour
                timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S.000Z')
                
            # Add trace to the event log
            eventlog.traces.append(trace)

        return eventlog

    @staticmethod
    def from_pm4py(pm4py_eventlog):
        """
        Convert a PM4Py event log to the custom event log.

        Parameters:
        -----------
        pm4py_eventlog : pm4py.objects.log.log.EventLog
            The PM4Py event log to convert.

        Returns:
        --------
        EventLog
            The custom event log.
        """
        eventlog = EventLog()  # Create an empty custom event log

        # Iterate through each trace in the PM4Py event log
        for pm4py_trace in pm4py_eventlog:
            trace_id = pm4py_trace.attributes.get("concept:name", "")
            trace_attributes = dict(pm4py_trace.attributes)

            # Create a new custom Trace object
            trace = Trace(trace_id, trace_attributes)

            # Iterate through each event in the PM4Py trace
            for pm4py_event in pm4py_trace:
                activity = pm4py_event.get("concept:name", "")
                timestamp = pm4py_event.get("time:timestamp", "")
                event_attributes = {k: v for k, v in pm4py_event.items() if k not in ["concept:name", "time:timestamp"]}

                # Create a new custom Event object
                event = Event(activity, timestamp, event_attributes)
                trace.add_event(event)

            # Add the trace to the custom event log
            eventlog.traces.append(trace)

        return eventlog

    def to_xes(self, xes_file: str):
        """
        Save the event log to an XES file.

        Parameters:
        -----------
        xes_file : str
            The path to the XES file to save.
        """
        # Create the root element for the XES log
        log = ET.Element("log", attrib={
            "xes.version": "1.0",
            "xes.features": "",
            "openxes.version": "1.0",
            "xmlns": "http://www.xes-standard.org/"
        })
        
        # Iterate through the traces and build their XML structure
        for trace in self.traces:
            trace_elem = ET.SubElement(log, "trace")
            
            # Add trace attributes (e.g., trace id)
            ET.SubElement(trace_elem, "string", key="concept:name", value=trace.trace_id)
            for key, value in trace.attributes.items():
                ET.SubElement(trace_elem, "string", key=key, value=value)

            # Add events within the trace
            for event in trace.events:
                event_elem = ET.SubElement(trace_elem, "event")
                
                # Add event attributes (e.g., activity and timestamp)
                ET.SubElement(event_elem, "string", key="concept:name", value=event.activity)
                ET.SubElement(event_elem, "date", key="time:timestamp", value=event.timestamp)
                for key, value in event.attributes.items():
                    if isinstance(value, str):  # Save as string
                        ET.SubElement(event_elem, "string", key=key, value=value)
                    else:  # Save other data types, adjust if needed
                        ET.SubElement(event_elem, "string", key=key, value=str(value))

        # Write the XML tree to the file
        tree = ET.ElementTree(log)
        tree.write(xes_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    def __repr__(self):
        """
        Provide a detailed representation of the event log, showing all traces and their events.
        """
        repr_str = f"EventLog with {len(self.traces)} traces:\n"
        for trace in self.traces:
            repr_str += f"Trace ID: {trace.trace_id}, Attributes: {trace.attributes}\n"
            for event in trace.events:
                repr_str += f"  Event: Activity={event.activity}, Timestamp={event.timestamp}, Attributes={event.attributes}\n"
        return repr_str
    
    def __len__(self):
        """
        Return the number of traces in the event log.
        """
        return len(self.traces)

    def get_trace_by_id(self, trace_id: str):
        """
        Retrieve a trace by its trace ID.

        Parameters:
        -----------
        trace_id : str
            The ID of the trace to retrieve.

        Returns:
        --------
        Trace or None
            The trace with the given trace ID, or None if not found.
        """
        for trace in self.traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    def to_pm4py(self):
        """
        Convert the custom event log to a PM4Py event log.

        Returns:
        --------
        pm4py.objects.log.log.EventLog
            The PM4Py event log.
        """
        pm4py_event_log = PM4PyEventLog()  # Create an empty PM4Py event log

        # Iterate through each trace in the custom event log
        for trace in self.traces:
            pm4py_trace = PM4PyTrace()  # Create an empty PM4Py trace

            # Add trace attributes (e.g., trace_id)
            pm4py_trace.attributes["concept:name"] = trace.trace_id
            for key, value in trace.attributes.items():
                pm4py_trace.attributes[key] = value

            # Add events to the PM4Py trace
            for event in trace.events:
                pm4py_event = PM4PyEvent()
                pm4py_event["concept:name"] = event.activity
                pm4py_event["time:timestamp"] = event.timestamp  # Timestamps must be datetime objects for PM4Py
                for key, value in event.attributes.items():
                    pm4py_event[key] = value

                pm4py_trace.append(pm4py_event)  # Append the event to the PM4Py trace

            # Append the trace to the PM4Py event log
            pm4py_event_log.append(pm4py_trace)

        return pm4py_event_log

    def get_footprint_matrix(self, length: int = 1) -> dict:
        """
        Calculate the footprint matrix for the event log considering eventual succession
        within a specified length. Returns a dictionary in the form:
        {('B', 'A'): '>', ('B', 'C'): '||'...}

        Parameters:
        ----------
        length : int
            The maximum number of steps allowed between two activities to consider them in succession.

        Returns:
        -------
        dict
            A dictionary where keys are tuples representing pairs of activities ('A', 'B')
            and values are the relations ('>', '||', '#').
        """
        # Step 1: Initialize a dictionary for direct succession relationships with eventual following
        all_activities = self.unique_activities()
        footprint_matrix = {}

        # Step 2: Fill the footprint matrix based on eventual succession relationships
        for activity_a in all_activities:
            for activity_b in all_activities:
                pair_key = (activity_a, activity_b)

                if self.does_eventually_follows(activity_a, activity_b, length):
                    if self.does_eventually_follows(activity_b, activity_a, length):
                        footprint_matrix[pair_key] = '||'  # Parallel relation
                    else:
                        footprint_matrix[pair_key] = '>'  # A → B (causal relation)
                elif self.does_eventually_follows(activity_b, activity_a, length):
                    footprint_matrix[pair_key] = '<'  # B → A (reverse causal relation)
                else:
                    footprint_matrix[pair_key] = '#'  # No direct relation

        return footprint_matrix
    
    def does_eventually_follows(self, activity_a: str, activity_b: str, length: int = 1) -> bool:
        """
        Check if `activity_a` is eventually followed by `activity_b` within `length` steps
        in any trace in the event log.

        Parameters:
        ----------
        activity_a : str
            The activity to check if it is followed by `activity_b`.
        activity_b : str
            The activity that should follow `activity_a`.
        length : int
            The maximum number of steps allowed between `activity_a` and `activity_b`.

        Returns:
        -------
        bool
            True if `activity_a` is eventually followed by `activity_b` within `length` steps
            in any trace, False otherwise.
        """
        for trace in self.traces:
            for i, event in enumerate(trace.events):
                if event.activity == activity_a:
                    # Check if `activity_b` appears within the next `length` events
                    for j in range(1, length + 1):
                        if i + j < len(trace.events) and trace.events[i + j].activity == activity_b:
                            return True
        return False
    
    def unique_activities(self):
        """
        Get the set of unique activities in the event log.
        """
        if self._unique_activities is not None:
            return self._unique_activities
        
        activities = set()
        for trace in self.traces:
            for event in trace.events:
                activities.add(event.activity)
        
        return activities
    
    def set_unique_activities(self, activities):
        """
        Set the unique activities of the event log.
        """
        self._unique_activities = activities
        
    def set_eventlog_name(self, name):
        """
        Set the name of the event log.
        """
        self.name = name
    
    def to_fast_token_based_replay(self):
        """
        Converts the Python EventLog object to a FastTokenBasedReplay.EventLog object.
        """
        # Create an empty FastTokenBasedReplay EventLog object
        event_log_c = FastTokenBasedReplay.EventLog()

        # Convert traces
        for trace in self.traces:
            # Create a corresponding FastTokenBasedReplay Trace object
            fast_trace = FastTokenBasedReplay.Trace(trace.trace_id, trace.attributes)

            # Convert events within the trace
            for event in trace.events:
                # Assuming FastTokenBasedReplay Event object has similar attributes
                fast_event = FastTokenBasedReplay.Event(event.activity, str(event.timestamp), event.attributes)
                fast_trace.add_event(fast_event)

            # Add the converted trace to the FastTokenBasedReplay EventLog
            event_log_c.add_trace(fast_trace)

        return event_log_c