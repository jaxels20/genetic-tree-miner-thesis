import os
import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer

base_dir = os.path.dirname(__file__)
pnml_path = os.path.join(base_dir, 'pnml_models/Nasa.pnml')

# Read the Petri net from PNML
net, im, fm = pm4py.read_pnml(pnml_path)

# Visualize the Petri net
gviz = pn_visualizer.apply(net, im, fm)
pn_visualizer.view(gviz)
