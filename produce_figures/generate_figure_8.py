
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
        "triangle-left", "triangle-right", "star", "hexagram", "hourglass", "arrow", "bowtie",
    ]
    # Add traces for each dataset
    for i, dataset in enumerate(df['dataset'].unique()):
        subset = df[df['dataset'] == dataset]
        fig.add_trace(go.Scatter(
            x=subset['percentage_of_log'],
            y=subset['objective_fitness'],
            mode='lines+markers',
            name=dataset,
            line=dict(color=colors[i]),
            marker=dict(
                symbol=marker_symbols[i],
                size=8,
                color=colors[i]
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
    
    # Write the figure to a PDF file
    fig.write_image(OUTPUT_FILE, format='pdf')


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv(INPUT_FILE)

    # Produce the figure
    produce_figure(df)
        