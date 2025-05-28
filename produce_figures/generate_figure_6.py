import os
import plotly.graph_objects as go
import pandas as pd

INPUT_DIR = './data/figure_6'
OUTPUT_DIR = "./figures/figure_6"

def visualize_figure(df, dataset):
    fig = go.Figure()
    
    # Colorblind-friendly, muted palette
    colors = {
        'Replay Fitness': '#6A5ACD',         # Slate Blue
        'Precision': '#3CB371',             # Medium Sea Green
        'Simplicity': '#DAA520',            # Goldenrod
        'Refined Simplicity': '#FF6347'     # Tomato
    }

    # Add line for objective fitness (neutral gray line)
    fig.add_trace(go.Scatter(
        x=df['generation'],
        y=df['objective_fitness'],
        mode='lines',
        name='Objective Fitness',
        line=dict(color='gray', width=4),
        marker=dict(color='gray')
    ))

    # Add stacked bars for each metric with appropriate color
    fig.add_trace(go.Scatter(
        mode='lines',
        x=df['generation'],
        y=df['ftr_fitness'],
        marker=dict(color=colors['Replay Fitness'], line=dict(color=colors['Replay Fitness'], width=0)),
        name='Replay Fitness'
    ))
    fig.add_trace(go.Scatter(
        mode='lines',
        x=df['generation'],
        y=df['ftr_precision'],
        marker=dict(color=colors['Precision'], line=dict(color=colors['Precision'], width=0)),
        name='Precision'
    ))
    fig.add_trace(go.Scatter(
        mode='lines',
        x=df['generation'],
        y=df['simplicity'],
        marker=dict(color=colors['Simplicity'], line=dict(width=0)),
        name='Simplicity'
    ))
    fig.add_trace(go.Scatter(
        mode='lines',
        x=df['generation'],
        y=df['refined_simplicity'],
        marker=dict(color=colors['Refined Simplicity'], line=dict(width=0)),
        name='Refined Simplicity'
    ))


    # Update layout for stacked bars and dual y-axis
    fig.update_layout(
        # barmode='stack',
        template='simple_white',
        xaxis_title='Generation',
        yaxis=dict(
            title='Objective Fitness',
            rangemode='tozero'
        ),
        # bargap=0,
        # bargroupgap=0,
        legend=dict(
            orientation='h',       # Horizontal orientation
            x=0,                   # Left aligned
            y=1.02,                # Slightly above the plot
            xanchor='left',
            yanchor='bottom'
        ),
        margin=dict(l=40, r=40, t=80, b=40),  # Increase top margin to make room for the legend
        height=500,
        width=800,
    )
    
    
    fig.write_image(f"{OUTPUT_DIR}/{dataset}.png", format='png')



if __name__ == "__main__":
    datasets = os.listdir(INPUT_DIR)
    
    for dataset in datasets:
        data_path = os.path.join(INPUT_DIR, dataset)     
        df = pd.read_csv(data_path)
        dataset_name = dataset.split('.')[0]
        visualize_figure(df, dataset_name)
        