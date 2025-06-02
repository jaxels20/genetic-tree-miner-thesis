
from plotly import graph_objects as go
import pandas as pd

INPUT_FILE = "./data/figure_8/data.csv"
OUTPUT_FILE = "./figures/figure_8.pdf"


def produce_figure(df):
    # Create figure
    fig = go.Figure()
    colors = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999",
        "#117733", "#332288", "#88CCEE", "#44AA99",
        "#661100", "#6699CC"
    ]
    marker_symbols = [
        "circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down",
        "triangle-left", "triangle-right", "star", "hexagram", "hourglass", "bowtie", "circle",
    ]
    datasets = ['2013-cp', '2013-op', '2013-i', 'RTF', '2012', 'Sepsis', '2020-rfp', '2020-id', '2020-dd', '2017', '2020-ptc', '2019', '2020-pl']
    
    color_map = {
        ds: colors[i] for i, ds in enumerate(datasets)
    }
    marker_map = {
        ds: marker_symbols[i] for i, ds in enumerate(datasets)
    }
        # Step 1: Compute max fitness per dataset
    dataset_order = (
        df.groupby("dataset")["objective_fitness"]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    
    df["percentage_of_log"] = df["percentage_of_log"] * 100  # Convert to percentage

    # Add traces for each dataset
    for i, dataset in enumerate(dataset_order):
        subset = df[df['dataset'] == dataset]
        fig.add_trace(go.Scatter(
            x=subset['percentage_of_log'],
            y=subset['objective_fitness'],
            mode='lines+markers',
            name=dataset,
            line=dict(color=color_map[dataset]),
            marker=dict(
                symbol=marker_map[dataset],
                size=8,
                color=color_map[dataset],
            )
        ))

    # Update layout
    fig.update_layout(
        xaxis_title="Percentage of Log",
        yaxis_title="Objective Fitness",
        legend_title="Dataset",
        template="simple_white",
        font=dict(family='Times New Roman', size=20),
        margin=dict(l=0, r=0, t=10, b=120),
        legend=dict(
            font=dict(size=17)
        )  
    )
    fig.update_yaxes(range=[50, 100])  # Set y-axis range from 0.5 to 1.0
    
    # Add percentage symbol to x-axis ticks
    fig.update_xaxes(tickvals=[0, 20, 40, 60, 80, 100], ticktext=["0%", "20%", "40%", "60%", "80%", "100%"])
    
    # Write the figure to a PDF file
    fig.write_image(OUTPUT_FILE, format='pdf')


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv(INPUT_FILE)
    df['objective_fitness'] = df['objective_fitness'] * 100

    # Produce the figure
    produce_figure(df)
        