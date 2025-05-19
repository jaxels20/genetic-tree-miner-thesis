import os
import pickle
import plotly.graph_objects as go
import pandas as pd
from plotly.colors import qualitative

OUTPUT_DIR = './experiment_5'

def visualize_paper_figure(input_dir, output_file_name):
    configurations = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    data = []
    
    for d in configurations:
        # Extract the folder name of d without the full path
        folder_name = os.path.basename(d)
        
        subfolders = [f.path for f in os.scandir(d) if f.is_dir()]
    
        for subfolder in subfolders:
            pkl_files = [f for f in os.listdir(subfolder + "/monitors") if f.endswith('.pkl')]
        
            for pkl_file in pkl_files:
                with open(os.path.join(subfolder, "monitors", pkl_file), 'rb') as f:
                    dataset_name, method_name, result_dict = pickle.load(f)                
                
                generations = list(result_dict.keys())
                fitness_values = list(result_dict.values())
                
                for i in range(len(generations)):
                    data.append(
                        {
                            "Generation": generations[i],
                            "Fitness": fitness_values[i],
                            "Dataset": dataset_name,
                            "Line Type": folder_name
                        }
                    )

    df = pd.DataFrame(data)
    
    # group by generation and line type and mean fitness
    df = df.groupby(["Line Type", "Generation"], as_index=False).mean(numeric_only=True)

    print(df)
    # 1) build a color‐map for each Line Type
    line_types = df["Line Type"].unique()
    colorblind_colors = ["grey", "lightblue"]
    color_map = {
        lt: colorblind_colors[i % len(colorblind_colors)]
        for i, lt in enumerate(line_types)
    }  
    ## 2) create the figure and add one trace per (Dataset, Line Type)
    fig = go.Figure()
    seen = set()

    for lt in line_types:
        # filter the dataframe for the current Line Type
        grp = df[df["Line Type"] == lt]
        # add a trace for each Line Type
        fig.add_trace(go.Scatter(
            x=grp["Generation"],
            y=grp["Fitness"],
            mode="lines",
            name=lt,                    # legend entry label
            legendgroup=lt,             # group traces of same Line Type
            showlegend=(lt not in seen),# only show legend once per Line Type
            line=dict(color=color_map[lt])
        ))
        seen.add(lt)

    # 3) Update layout
    fig.update_layout(
        title=None,
        xaxis_title="Generation",
        yaxis_title="Objective Fitness",
        font=dict(family='Times', size=16),
        margin=dict(l=60, r=30, t=50, b=120),
        width=900,
        height=600,
        template='simple_white',
        legend=dict(
            font=dict(size=14, family="Times New Roman"),
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )

    fig.update_yaxes(range=[20, 100])
    
    # write the file to the output directory
    fig.write_image(f"{OUTPUT_DIR}/{output_file_name}.pdf")
    
    
if __name__ == "__main__":
    visualize_paper_figure("./experiment_5", "figure_5")
    