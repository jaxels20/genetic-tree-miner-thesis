import os
import pm4py

# Define paths
base_dir = os.path.dirname(__file__)
bpmn_dir = os.path.join(base_dir, 'bpmn_models')
pnml_dir = os.path.join(base_dir, 'pnml_models')

# Ensure output directory exists
os.makedirs(pnml_dir, exist_ok=True)

# Iterate through all BPMN files in the input directory
for filename in os.listdir(bpmn_dir):
    if filename.endswith('.bpmn'):
        bpmn_path = os.path.join(bpmn_dir, filename)
        pnml_filename = os.path.splitext(filename)[0] + '.pnml'
        pnml_path = os.path.join(pnml_dir, pnml_filename)

        try:
            print(f"Converting: {filename}")
            # Read the BPMN model
            bpmn_model = pm4py.read_bpmn(bpmn_path)

            # Convert to Petri net
            net, im, fm = pm4py.convert_to_petri_net(bpmn_model)

            # Write to PNML file
            pm4py.write_pnml(net, im, fm, pnml_path)
            print(f"Saved: {pnml_path}")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
