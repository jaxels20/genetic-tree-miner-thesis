import os
import pickle
import plotly.graph_objects as go
import pandas as pd

INPUT_DIR_RANDOM = './data/figure_5d'
INPUT_DIR_INDUCTIVE = './data/figure_5c/'
OUTPUT_FILE = "./figures/figure_5d.pdf"

def visualize_paper_figure():
    data = []
    input_dirs = {'Random Tree Generator': INPUT_DIR_RANDOM, 'IM Tree Generator': INPUT_DIR_INDUCTIVE}
    
    for name_dir, input_dir in input_dirs.items():
        for root, subdirs, files in os.walk(input_dir):
            for pkl_file in files:
                if not pkl_file.endswith('.pkl'):
                    continue
                
                with open(os.path.join(root, pkl_file), 'rb') as f:
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
                            "Line Type": name_dir
                        }
                    )

    df = pd.DataFrame(data)
    
    # Aggregate 
    aggregated = df.groupby(["Line Type", "Generation"], as_index=False).mean(numeric_only=True)
    aggregated["Dataset"] = "Aggregated"

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
    