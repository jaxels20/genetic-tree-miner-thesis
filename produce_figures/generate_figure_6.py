import os
import plotly.graph_objects as go
import pandas as pd

INPUT_DIR = './data/figure_6'
OUTPUT_DIR = "./figures"

def visualize_figure():    
    colors = [
        "#332288","#56B4E9","#E69F00","#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999",
        "#117733", "#88CCEE", "#44AA99",
        "#6699CC", "#661100"
    ]
    
    symbols = {
        'ftr_precision': 'circle',
        'ftr_fitness': 'square',
        'simplicity': 'diamond',
        'refined_simplicity': 'x'
    }
    
    weights = {
        'ftr_precision': 30,
        'ftr_fitness': 50,
        'simplicity': 10,
        'refined_simplicity': 10
    }
    
    rename_dict = {
        'ftr_precision': 'Precision',
        'ftr_fitness': 'Fitness',
        'simplicity': 'Simplicity',
        'refined_simplicity': 'Refined Simplicity'
    }

    fig = go.Figure()
    for i, col in enumerate(['ftr_precision', 'ftr_fitness', 'simplicity', 'refined_simplicity']):
        concat_df = pd.DataFrame()
        for dataset in datasets:
            data_path = os.path.join(INPUT_DIR, dataset)     
            df = pd.read_csv(data_path)
            concat_df = pd.concat([concat_df, df], ignore_index=True)
            
        df = concat_df.groupby('generation').mean().reset_index()
        
        #Multiply the values by 100
        df[col] = df[col] * 100
        
        # dataset_name = dataset.split('.')[0]
        fig.add_trace(go.Scatter(
            mode='lines',
            x=df['generation'],
            y=df[col] / weights[col],
            name=rename_dict[col],
            line=dict(color=colors[i], width=2),
            showlegend=False
        ))
        step = 10
        marker_df = df.iloc[::step] 
        fig.add_trace(go.Scatter(
            mode='markers',
            x=marker_df['generation'],
            y=marker_df[col] / weights[col],
            name=rename_dict[col],
            marker=dict(
                    color=colors[i],
                    symbol=symbols[col],
                    size=8
                ),
            showlegend=True
        ))
        
    # Update layout for stacked bars and dual y-axis
    fig.update_layout(
        # barmode='stack',
        template='simple_white',
        xaxis_title='Generation',
        font=dict(
            size=20,
            family="Times New Roman"
        ),
        yaxis=dict(
            title='Score',
            rangemode='tozero'
        ),
        legend=dict(
            font=dict(size=20, family="Times New Roman"),
            orientation="v",
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        ),
        margin=dict(l=0, r=0, t=0, b=120),
    )
    
    fig.update_xaxes(range=[0, 300])
    fig.update_yaxes(range=[50, 100])
    fig.write_image(f"{OUTPUT_DIR}/figure_6.pdf")
    

if __name__ == "__main__":
    datasets = os.listdir(INPUT_DIR)
    visualize_figure()