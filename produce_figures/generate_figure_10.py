import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_DATA = "./data/figure_10/data.csv"
OUTPUT = "./figures/figure_10.pdf"

def generate_plot(df):
    fig = go.Figure()
    colors = {
        "Precision": "#332288",
        "Fitness": "#56B4E9",
        "Simplicity": "#E69F00",
        "Refined Simplicity": "#009E73"
        
    }
        
    symbols = {
        'Precision': 'circle',
        'Fitness': 'square',
        'Simplicity': 'diamond',
        'Refined Simplicity': 'x'
    }
    
    df.replace({'metric': {
        'ftr_fitness': 'Fitness', 
        'ftr_precision': 'Precision', 
        'simplicity': 'Simplicity', 
        'refined_simplicity': 'Refined Simplicity',}
    }, inplace=True)
    
    metric_list = ["Precision", "Fitness", "Simplicity", "Refined Simplicity"]

    for i, metric in enumerate(metric_list):
        metric_df = df[df["metric"] == metric]
        fig.add_trace(go.Scatter(
            x=metric_df["weight_share"],
            y=metric_df["value"],
            mode='lines+markers',
            name=metric,
            marker=dict(
                color=colors[metric],
                symbol=symbols[metric],
                size=8
            ),
            line=dict(color=colors[metric], width=2)
        ))
            

    # Clean white theme and layout
    fig.update_layout(
        template='simple_white',
        xaxis_title='Weight Share',
        yaxis_title='Score',
        font=dict(
            size=20,
            family="Times New Roman"
        ),
        margin=dict(l=0, r=0, t=0, b=120),
        legend=dict(
            font=dict(size=20, family="Time New Roman"),
            orientation="v",
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        ),
    )
    fig.update_yaxes(
        range=[80, 100],
    )
    # Save one combined figure
    fig.write_image(OUTPUT)
    
if __name__ == "__main__":
    df = pd.read_csv(INPUT_DATA)
    
    df.drop(columns=['dataset'], inplace=True)
    df = df.groupby(['weight_share', 'metric']).mean().reset_index()
    
    df['value'] = df['value'] * 100
    
    generate_plot(df)