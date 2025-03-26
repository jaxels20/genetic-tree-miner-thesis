import pm4py
import os
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer

def xes_to_petri_and_tree(xes_file):
    # Ensure the file exists
    if not os.path.exists(xes_file):
        print(f"Error: The file {xes_file} does not exist.")
        return
    
    # Read the event log
    log = pm4py.read_xes(xes_file)
    
    # Discover the Petri net using the Inductive Miner
    net, im, fm = pm4py.discover_petri_net_inductive(log)
    
    # Visualize and save the Petri net
    petri_gviz = pn_visualizer.apply(net, im, fm)
    petri_output_file = os.path.join(os.path.dirname(xes_file), "output_petri.png")
    pn_visualizer.save(petri_gviz, petri_output_file)
    print(f"Petri net saved to {petri_output_file}")
    
    # Discover the process tree using the Inductive Miner
    process_tree = pm4py.discover_process_tree_inductive(log)
    
    # Visualize and save the process tree
    tree_gviz = pt_visualizer.apply(process_tree)
    tree_output_file = os.path.join(os.path.dirname(xes_file), "output_tree.png")
    pt_visualizer.save(tree_gviz, tree_output_file)
    print(f"Process tree saved to {tree_output_file}")

def process_all_xes_files(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".xes"):
                xes_file_path = os.path.join(subdir, file)
                print(f"Processing {xes_file_path}...")
                xes_to_petri_and_tree(xes_file_path)

# Define the root directory containing subfolders with .xes files
root_directory = "real_life_datasets"
process_all_xes_files(root_directory)