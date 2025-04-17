from src.PetriNet import PetriNet
from src.EventLog import EventLog
import os
from multiprocessing import Pool


class FileLoader:
    """
    A utility class to load PetriNet and EventLog objects in batches or as a whole,
    supporting multiprocessing for efficiency.
    """

    def __init__(self, cpu_count: int = 1):
        pass

    @staticmethod
    def load_petrinet(file_path: str):
        """
        Helper function to load a single PetriNet object from a file.
        Extracts the ID and returns a tuple of (id, PetriNet object).
        Supports both .ptml and .pnml files.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [".ptml", ".pnml"]:
            return None  # Skip unsupported files

        # Determine how to load the PetriNet based on extension
        if ext == ".ptml":
            pn = PetriNet.from_ptml(file_path)
        else:  # ext == ".pnml"
            pn = PetriNet.from_pnml(file_path)

        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]

        return name_without_ext, pn

    @staticmethod
    def load_eventlog(file_path: str):
        """
        Helper function to load a single EventLog object from a file.
        Extracts the ID and returns a tuple of (id, EventLog object).
        """
        el = EventLog.load_xes(file_path)
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        el.set_eventlog_name(name_without_ext)
        return el