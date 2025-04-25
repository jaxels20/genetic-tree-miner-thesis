import os
from src.FileLoader import FileLoader
from src.Objective import Objective
from src.PetriNet import PetriNet

INPUT_DIR = "./real_life_datasets/"
INPUT_PN_DIR = "./splitminer/pnml_models/"
OUTPUT_DIR = "./splitminer/"

if __name__ == "__main__":
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()

    for dataset_dir in dataset_dirs:
        if dataset_dir in ["2013-op", "2013-cp", "2013-i", "2017", "2019", "2020-pl"]:  # delete later
            continue
        xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
        if len(xes_file) == 0:
            continue
        elif len(xes_file) == 1:
            #loaded_log = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")
            pn = PetriNet.from_pnml(f"{INPUT_PN_DIR}{dataset_dir}.pnml")
            print(pn)
            break
            #objective = Objective()
            
           # objective.set_event_log(loaded_log)
        else:
            raise ValueError("More than one xes file in the directory")
    