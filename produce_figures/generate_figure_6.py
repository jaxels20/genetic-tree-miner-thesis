import os
import plotly.graph_objects as go
import pandas as pd

INPUT_DIR = './data/figure_6'
OUTPUT_DIR = "./figures/figure_6"

def visualize_figure(df, dataset_name):
    fig = go.Figure()
    
    # Colorblind-friendly, muted palette
    colors = {
        'Replay Fitness': '#4477AA',
        'Precision': '#EE6677',
        'Simplicity': '#228833',
        'Refined Simplicity': '#AA3377'
    }

    # Add stacked bars for each metric with appropriate color
    fig.add_trace(go.Bar(
        x=df['generation'],
        y=df['ftr_fitness'],
        marker=dict(color=colors['Replay Fitness'], line=dict(color=colors['Replay Fitness'], width=0)),
        name='Replay Fitness'
    ))
    fig.add_trace(go.Bar(
        x=df['generation'],
        y=df['ftr_precision'],
        marker=dict(color=colors['Precision'], line=dict(color=colors['Precision'], width=0)),
        name='Precision'
    ))
    fig.add_trace(go.Bar(
        x=df['generation'],
        y=df['simplicity'],
        marker=dict(color=colors['Simplicity'], line=dict(width=0)),
        name='Simplicity'
    ))
    fig.add_trace(go.Bar(
        x=df['generation'],
        y=df['refined_simplicity'],
        marker=dict(color=colors['Refined Simplicity'], line=dict(width=0)),
        name='Refined Simplicity'
    ))

    # Add line for objective fitness (neutral gray line)
    fig.add_trace(go.Scatter(
        x=df['generation'],
        y=df['objective_fitness'],
        mode='lines+markers',
        name='Objective Fitness',
        line=dict(color='gray', width=2),
        marker=dict(color='gray')
    ))

    # Update layout for stacked bars and dual y-axis
    fig.update_layout(
        barmode='stack',
        template='simple_white',
        xaxis_title='Generation',
        yaxis=dict(
            title='Stacked Metric Scores',
            rangemode='tozero'
        ),
        bargap=0,
        bargroupgap=0,
        legend=dict(x=1.05, y=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=500
    )
    
    fig.write_image(f"{OUTPUT_DIR}/figure_6_{dataset_name}.png", format='png')



if __name__ == "__main__":
    datasets = os.listdir(INPUT_DIR)
    
    for dataset in datasets:
        data_path = os.path.join(INPUT_DIR, dataset, "data.csv")     
        df = pd.read_csv(data_path)
        visualize_figure(df, dataset)
        