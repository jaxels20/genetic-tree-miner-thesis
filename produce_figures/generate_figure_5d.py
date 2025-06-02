import os
import pickle
import plotly.graph_objects as go
import pandas as pd

INPUT_DIR = './data/figure_5d/'
OUTPUT_FILE = "./figures/figure_5d.pdf"

def visualize_paper_figure():
    experiments = [f.path for f in os.scandir(INPUT_DIR) if f.is_dir()]
    data = []
    
    for d in experiments:
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
                
                generations = generations[::5]
                fitness_values = fitness_values[::5]
                
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
    df.replace({"Line Type": {"inductive_tree_generator": "IM Tree Generator", "random_tree_generator": "Random Tree Generator"}}, inplace=True)
    
    # Aggregate 
    aggregated = df.groupby(["Line Type", "Generation"], as_index=False).mean(numeric_only=True)
    aggregated["Dataset"] = "Aggregated"
    
    include_datasets = ["2019", "2017", "2020-pl"]
    df = df[df["Dataset"].isin(include_datasets)]

    # 1) build a color‐map for each Line Type
    line_types = df["Line Type"].unique()
    colorblind_colors = [
        "black", "red"
    ]
    color_map = {
        lt: colorblind_colors[i % len(colorblind_colors)]
        for i, lt in enumerate(line_types)
    }
    
    marker_symbols = [
        "x", "circle",
    ]
    marker_map = {
        lt: marker_symbols[i % len(marker_symbols)]
        for i, lt in enumerate(line_types)
    }
    step = 2
      
    # 2) create the figure and add one trace per (Dataset, Line Type)
    fig = go.Figure()

    # Add the mean line for the aggregated data
    for i, lt in enumerate(line_types):
        grp_lt = aggregated[aggregated["Line Type"] == lt]
        fig.add_trace(go.Scatter(
            x=grp_lt["Generation"],
            y=grp_lt["Fitness"],
            mode="lines",
            name=f"{lt}",
            legendgroup=lt,
            showlegend=False,
            line=dict(color=color_map[lt], dash="solid", width=2)
        ))
        marker_df = grp_lt.iloc[::step]  # start from 'offset', then every 'step'
        fig.add_trace(go.Scatter(
            x=marker_df["Generation"],
            y=marker_df["Fitness"],
            mode="markers",
            name=f"{lt}",
            legendgroup=lt,
            showlegend=True,
            marker=dict(
                symbol=marker_map[lt],
                size=8,
                color=color_map[lt]
            )
        ))

    # 3) Update layout
    fig.update_layout(
        title=None,
        xaxis_title="Generation",
        yaxis_title="Objective Fitness",
        font=dict(family='Times New Roman', size=20),
        margin=dict(l=0, r=0, t=0, b=120),
        template='simple_white',
        legend=dict(
            font=dict(size=18, family="Times New Roman"),
            orientation="v",
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )

    fig.update_yaxes(range=[55, 100])
    
    # write the file to the output directory
    fig.write_image(OUTPUT_FILE)
    

if __name__ == "__main__":
    visualize_paper_figure()
    