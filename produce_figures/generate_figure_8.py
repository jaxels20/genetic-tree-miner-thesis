
from plotly import graph_objects as go
import pandas as pd

INPUT_FILE = "./data/figure_8/data.csv"
OUTPUT_FILE = "./figures/figure_8.pdf"


def produce_figure(df):
    # Create figure
    fig = go.Figure()

    # Add traces for each dataset
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        fig.add_trace(go.Scatter(
            x=subset['percentage_of_log'],
            y=subset['objective_fitness'],
            mode='lines+markers',
            name=dataset
        ))

    # Update layout
    fig.update_layout(
        xaxis_title="Percentage of Log",
        yaxis_title="Objective Fitness",
        legend_title="Dataset",
        template="plotly_white"
    )
    
    # Write the figure to a PDF file
    fig.write_image(OUTPUT_FILE, format='pdf')


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv(INPUT_FILE)

    # Produce the figure
    produce_figure(df)
        