""" This module contains the implementation of the discovery algorithms that are used to discover a Petri net from an event log. """

from EventLog import EventLog
from PetriNet import PetriNet
from pm4py.algo.discovery.alpha.algorithm import apply as pm4py_alpha_miner
from pm4py.algo.discovery.heuristics.algorithm import apply as pm4py_heuristic_miner
from pm4py.algo.discovery.inductive.algorithm import apply as pm4py_inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter


class Discovery:
    @staticmethod
    def alpha_miner(event_log: EventLog) -> PetriNet:
        """
        A wrapper for the alpha miner algorithm that is implemented in the pm4py library.
        """
        pm4py_event_log = event_log.to_pm4py()
        pm4py_net, pm4py_initial_marking, pm4py_final_marking = pm4py_alpha_miner(pm4py_event_log)
        
        net = PetriNet.from_pm4py(pm4py_net)
        return net

    @staticmethod
    def heuristic_miner(event_log: EventLog) -> PetriNet:
        """
        A wrapper for the heuristic miner algorithm that is implemented in the pm4py library.
        """
        pm4py_event_log = event_log.to_pm4py()
        pm4py_net, pm4py_initial_marking, pm4py_final_marking = pm4py_heuristic_miner(pm4py_event_log)
        
        net = PetriNet.from_pm4py(pm4py_net)
        return net

    @staticmethod
    def inductive_miner(event_log: EventLog) -> PetriNet:
        """
        A wrapper for the inductive miner algorithm that is implemented in the pm4py library.
        """
        pm4py_event_log = event_log.to_pm4py()
        pm4py_process_tree = pm4py_inductive_miner(pm4py_event_log)
        pm4py_net, _, _ = pt_converter.apply(pm4py_process_tree)
        
        net = PetriNet.from_pm4py(pm4py_net)
        return net


    # Map method names to static methods
    methods = {
        "Alpha": alpha_miner,
        "Heuristic": heuristic_miner,
        "Inductive": inductive_miner,
    }

    @classmethod
    def run_discovery(cls, method_name: str, event_log: EventLog, **kwargs) -> PetriNet:
        """
        Runs the specified discovery method based on method_name.
        
        Parameters:
        - method_name (str): The name of the method to run (e.g., "alpha", "heuristic", "inductive", "aau_miner", "gnn_miner").
        - event_log (EventLog): The event log to be passed to the discovery method.
        - **kwargs: Additional arguments to be passed to the method, such as model_path for GNN_miner.
        
        Returns:
        - PetriNet: The discovered Petri net.
        """
        method = cls.methods.get(method_name)
        if method is None:
            raise ValueError(f"Discovery method '{method_name}' not found.")
        
        # Call the appropriate method, passing **kwargs for any extra arguments needed
        return method(event_log, **kwargs)
    
    
    
    