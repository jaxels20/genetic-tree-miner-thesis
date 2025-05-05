
import zss
from zss import Node
from src.ProcessTree import ProcessTree, Operator
from src.RandomTreeGenerator import BottomUpRandomBinaryGenerator, FootprintGuidedSequentialGenerator, InductiveNoiseInjectionGenerator, InductiveMinerGenerator
from src.EventLog import EventLog
from experiment_1 import load_hyperparameters_from_csv
from src.Objective import Objective
from src.FileLoader import FileLoader
from src.Discovery import Discovery
import os
import plotly.graph_objects as go
# el = EventLog.load_xes("./real_life_datasets/2013-cp/2013-cp.xes")

# generator1 = BottomUpRandomBinaryGenerator()
# generator2 = FootprintGuidedSequentialGenerator()
# generator3 = InductiveNoiseInjectionGenerator(0.05)
# generator4 = InductiveMinerGenerator()

# diversity = generator1.generate_population(el.unique_activities(), 100).diversity()
# print("BottomUpRandomBinaryGenerator diversity: ", diversity)

# diversity = generator2.generate_population(el, 100).diversity()
# print("FootprintGuidedSequentialGenerator diversity: ", diversity)

# diversity = generator3.generate_population(el, 100).diversity()
# print("InductiveNoiseInjectionGenerator diversity: ", diversity)

# diversity = generator4.generate_population(el, 100).diversity()
# print("InductiveMinerGenerator diversity: ", diversity)



INPUT_DIR = "./real_life_datasets/"
DATASET = "./real_life_datasets/2013-cp/2013-cp.xes"
TIME_LIMT = 5 * 60
STAGNATION_LIMIT = 50
BEST_PARAMS = "./best_parameters.csv"
NUM_SAMPLES = 20
OBJECTIVE_WEIGHTS = {
    "simplicity": 10,
    "refined_simplicity": 10,
    "ftr_fitness": 50,
    "ftr_precision": 30
}


def get_data():
    dataset_dirs = os.listdir(INPUT_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    loader = FileLoader()
    eventlogs = []
    
    best_hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)
    data = []

    for dataset_dir in dataset_dirs:
        # Assume only one file per directory
        if dataset_dir not in ["2013-cp"]:
            continue
          
        xes_file = [f for f in os.listdir(f"{INPUT_DIR}{dataset_dir}") if f.endswith(".xes")]
        if len(xes_file) == 0:
            continue
        elif len(xes_file) == 1:
            loaded_log = loader.load_eventlog(f"{INPUT_DIR}{dataset_dir}/{xes_file[0]}")
            eventlogs.append(loaded_log)
        else:
            raise ValueError("More than one xes file in the directory")
    
    
    
    for eventlog in eventlogs:
        
        best_hyper_parameters['random_creation_rate'] = 0.05
        best_hyper_parameters['elite_rate'] = 0.4
        best_hyper_parameters['tournament_rate'] = 0.55
        
        discovered_net, monitor = Discovery.genetic_algorithm(
            eventlog,
            time_limit=TIME_LIMT,
            **best_hyper_parameters,
            stagnation_limit=STAGNATION_LIMIT,
            percentage_of_log=0.05,
            return_monitor=True,
        )
        
        
        generations = monitor.generations
        diversities = [pop.diversity() for pop in monitor.populations]
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=generations, y=diversities, mode='lines+markers', name='Diversity'))
        
        
        # Create the experiment_6 directory if it doesn't exist
        if not os.path.exists("./experiment_6"):
            os.makedirs("./experiment_6")
        
        fig.write_image(f"./experiment_6/diversity_{eventlog.name}.png")
            
            
        
    
if __name__ == "__main__":
    get_data()

