import pandas as pd
import plotly.graph_objects as go

OUTPUT_PATH = "./experiment_2/"

# Parameters for generating the correctly formatted csv file
PRODUCE_DF = True
SOURCE_CSV = "./experiment_1/results_genetic_5_runs.csv"
DF_FILE_NAME = 'data.csv'

# Plot parameters
PLOT_DF = True
INPUT_DF = './experiment_2/data.csv'
PLOT_FILE_NAME = 'plot.pdf'

def produce_df(csv_file, df_output_file_name):
    # Load the data
    df = pd.read_csv(csv_file)
    
    # melt the file 
    df = df.melt(id_vars='Dataset', var_name='Metric', value_name='Score')
    df = df[df['Metric'] == 'Objective Fitness']
    df.sort_values(by=['Dataset', 'Metric'], inplace=True)
    
    # rename the dataset values
    df['Dataset'] = df['Dataset'].replace({
        '2019': '*2019',
        '2017': '*2017',
        '2020-id': '*2020-id',
        "2020-pl": "*2020-pl",
        "RTF": "*RTF",
        "2012": "*2012",
    })
    
    df.to_csv(OUTPUT_PATH + df_output_file_name, index=False)
        
def plot_df(df, plot_file_name):    
    fig = go.Figure()
    color = "lightgrey"

    # Compute IQR per Dataset
    iqr_df = (
        df.groupby('Dataset')['Score']
        .agg(lambda x: x.quantile(1) - x.quantile(0))
        .reset_index(name='IQR')
    )

    # Sort datasets by IQR
    sorted_datasets = iqr_df.sort_values('IQR')['Dataset'].tolist()

    fig.add_trace(go.Scatter(
        x=df['Dataset'],
        y=df['Score'],
        mode='markers',
        name='Objective Fitness',
        marker=dict(color=color, size=6, line=dict(width=1, color='black'))
    ))

    # Update x-axis order
    fig.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=sorted_datasets)
    )

    # Layout adjustments
    fig.update_layout(
        boxmode='group',  # group boxes of same x-axis value
        font=dict(family='Times', size=16),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            xanchor='center',
            y=1.05,        # slightly above the top of the plot
            x=0.5
        ),
        xaxis_title='Dataset',
        yaxis_title='Objective Fitness',
        margin=dict(l=60, r=30, t=50, b=120),
        template='simple_white',
        height=500,
        width=900
    )
    # set the y axisto 0 to 1
    fig.update_yaxes(range=[60, 100], dtick=10)
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45, tickmode='array', tickvals=sorted_datasets)
    
    # save the plot
    fig.write_image(OUTPUT_PATH + plot_file_name)
    

if __name__ == "__main__":
    if PRODUCE_DF:
        produce_df(csv_file=SOURCE_CSV, df_output_file_name=DF_FILE_NAME)
    
    if PLOT_DF:
        df = pd.read_csv(INPUT_DF)
        plot_df(df, plot_file_name=PLOT_FILE_NAME)

    