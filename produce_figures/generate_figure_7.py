import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd

INPUT_DATA = "./data/figure_7/data.csv"
OUTPUT_DIR = "./figures/figure_7/"

def create_plot(df):
    # Define the column used for coloring
    color_col = 'objective_fitness'

    # Drop it from the list of dimensions
    dimension_cols = [col for col in df.columns if col not in [color_col, 'dataset']]

    # Define custom axis limits and ticks
    custom_ylim =  {
        'random_creation_rate': [0, 1],
        'elite_rate': [0, 1],
        'population_size': [20, 70],
        'tournament_size': [0.1, 0.3],
        'log_filtering': [0.001, 0.200],
        'tournament_rate': [0, 1],
        'tournament_mutation_rate': [0, 1],
    }

    y_tick_vals = {
        'random_creation_rate': [x / 100 for x in range(0, 102, 10)],
        'elite_rate': [x / 100 for x in range(0, 102, 10)],
        'population_size': list(range(20, 75, 5)),
        'tournament_size': [x / 100 for x in range(10, 31, 2)],
        'log_filtering': [x / 1000 for x in range(1, 202, 20)],
        'tournament_rate': [x / 100 for x in range(0, 102, 10)],
        'tournament_mutation_rate': [x / 100 for x in range(0, 102, 10)],
    }

    # return unique values of specific column in df
    unique_datasets = df["dataset"].unique()
    
    for dataset in unique_datasets:
        # filter the dataframe for the current dataset
        df_dataset = df[df["dataset"] == dataset]
        
        # Build dimensions
        dimensions = []
        for col in dimension_cols:
            values = df_dataset[col].tolist()
            dim = dict(
                label=col.replace('_', ' ').title(),
                values=[x if not np.isnan(x) else None for x in values]
            )
            if col in custom_ylim:
                dim['range'] = custom_ylim[col]
                dim['tickvals'] = y_tick_vals[col]
            dimensions.append(dim)

        # Create parallel coordinates plot
        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=df_dataset[color_col],
                    colorscale='Viridis',
                    cmin=0.6,
                    cmax=1.0,
                    colorbar=dict(
                        title=dict(
                            text='Obj. Fitness', 
                            font=dict(size=16)
                        ),
                        tickfont=dict(size=16)
                    )
                ),
                dimensions=dimensions,
                labelside='bottom',
                labelangle=15
            )
        )

        # Layout settings
        fig.update_layout(
            font=dict(family='Times New Roman', size=18),
            margin=dict(l=80, r=50, t=10, b=50),
            template='simple_white'
        )

        # make a directory of output_dir if it does not exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # Save to file
        fig.write_image(f"{OUTPUT_DIR}/{dataset}.pdf")


import numpy as np
import os
import plotly.graph_objects as go

def create_combined_plot(df, OUTPUT_DIR):
    color_col = 'objective_fitness'

    # Dimensions = all except 'objective_fitness' and 'dataset'
    dimension_cols = [col for col in df.columns if col not in [color_col, 'dataset']]

    # Custom axis limits and ticks
    custom_ylim = {
        'random_creation_rate': [0, 1],
        'elite_rate': [0, 1],
        'population_size': [20, 70],
        'tournament_size': [0.1, 0.3],
        'log_filtering': [0.001, 0.200],
        'tournament_rate': [0, 1],
        'tournament_mutation_rate': [0, 1],
    }

    y_tick_vals = {
        'random_creation_rate': [x / 100 for x in range(0, 102, 10)],
        'elite_rate': [x / 100 for x in range(0, 102, 10)],
        'population_size': list(range(20, 75, 5)),
        'tournament_size': [x / 100 for x in range(10, 31, 2)],
        'log_filtering': [x / 1000 for x in range(1, 202, 20)],
        'tournament_rate': [x / 100 for x in range(0, 102, 10)],
        'tournament_mutation_rate': [x / 100 for x in range(0, 102, 10)],
    }

    # Build dimensions
    dimensions = []
    for col in dimension_cols:
        dim = dict(
            label=col.replace('_', ' ').title(),
            values=[x if not np.isnan(x) else None for x in df[col]]
        )
        if col in custom_ylim:
            dim['range'] = custom_ylim[col]
            dim['tickvals'] = y_tick_vals[col]
        dimensions.append(dim)
    
    # print("Dimensions:", len(dimensions[0]['values']))

    # Create the plot
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df[color_col],
                colorscale='Viridis',
                cmin=df[color_col].min(),
                cmax=df[color_col].max(),
                colorbar=dict(
                    title=dict(text='Objective Fitness', font=dict(size=16)),
                    tickfont=dict(size=16)
                )
            ),
            dimensions=dimensions,
            labelside='bottom',
            labelangle=15
        )
    )

    # Layout
    fig.update_layout(
        font=dict(family='Times New Roman', size=18),
        margin=dict(l=80, r=50, t=10, b=50),
        template='simple_white'
    )

    fig.write_image(f"{OUTPUT_DIR}/combined_plot.pdf")


def create_new_plot(df, baseline):
    fig = go.Figure()
    color = "lightgrey"
    
    df['objective_fitness'] = df['objective_fitness'] * 100  # Convert to percentage

    # Compute IQR per Dataset
    iqr_df = (
        df.groupby('dataset')['objective_fitness']
        .agg(lambda x: x.quantile(1) - x.quantile(0))
        .reset_index(name='IQR')
    )

    # Sort datasets by IQR
    sorted_datasets = iqr_df.sort_values('IQR')['dataset'].tolist()

    fig.add_trace(go.Scatter(
        x=df['dataset'],
        y=df['objective_fitness'],
        mode='markers',
        name='Objective Fitness',
        marker=dict(color=color, size=6, line=dict(width=1, color='black'))
    ))
    
    fig.add_trace(go.Scatter(
        x=baseline['dataset'],
        y=baseline['objective_fitness'] * 100,  # Convert to percentage
        mode='markers',
        name='Baseline',
        marker=dict(color='red', size=6, line=dict(width=1, color='black')),
    ))

    # Update x-axis order
    fig.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=sorted_datasets)
    )

    # Layout adjustments
    fig.update_layout(
        boxmode='group',  # group boxes of same x-axis value
        font=dict(family='Times New Roman', size=20),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            xanchor='center',
            y=1.05,        # slightly above the top of the plot
            x=0.5
        ),
        xaxis_title='Dataset',
        yaxis_title='Objective Fitness',
        margin=dict(l=0, r=0, t=0, b=150),
        template='simple_white',
    )
    # set the y axisto 0 to 1
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45, tickmode='array', tickvals=sorted_datasets)
    
    # save the plot
    fig.write_image(f"{OUTPUT_DIR}/new_plot.pdf")

    


if __name__ == "__main__":
    data = pd.read_csv(INPUT_DATA)
    
    baseline = pd.read_csv("./data/figure_7/baseline.csv")
    
    # for dataset in data['dataset'].unique():
    #     baseline_value = baseline[(baseline['dataset'] == dataset)]['objective_fitness'].values[0]
    #     data.loc[data['dataset'] == dataset, 'objective_fitness'] = ((data.loc[data['dataset'] == dataset, 'objective_fitness'] - baseline_value) / baseline_value) * 100
    
    #create_combined_plot(data, OUTPUT_DIR)
    #create_plot(data)
    create_new_plot(data, baseline)



