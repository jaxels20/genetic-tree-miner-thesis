import pandas
import os
from itertools import cycle
from src.Discovery import Discovery
from src.EventLog import EventLog 
from src.Evaluator import SingleEvaluator
import plotly.graph_objects as go
import plotly.express as px
from experiment_1 import load_hyperparameters_from_csv

BEST_PARAMS = "./best_parameters.csv"
DATASET_DIR = "./real_life_datasets/"
NUM_DATA_POINTS = 5
OBJECTIVE = {
    "simplicity": 10,
    "refined_simplicity": 10,
    "ftr_fitness": 50,
    "ftr_precision": 30
}

# TEST_DATASETS = ['2019', '2013-op', '2020-dd', '2020-ptc', "2020-rfp"]

def generate_data():
    # convert the hyper parameters to a normalize
    hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)
    
    dataset_dirs = os.listdir(DATASET_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{DATASET_DIR}{x}")]

    for dataset_dir in dataset_dirs:
        # Load the event log
        eventlog = EventLog.load_xes(f"{DATASET_DIR}{dataset_dir}/{dataset_dir}.xes")

        data = []
        for i in range(NUM_DATA_POINTS):
            discovered_net = Discovery.genetic_algorithm(
                eventlog,
                time_limit=60*5,
                stagnation_limit=50,
                **hyper_parameters,
                percentage_of_log=0.05,
            )
            
            evaluator = SingleEvaluator(
                discovered_net,
                eventlog
            )
            
            # Get the evaluation metrics
            metrics = {}
            metrics['dataset'] = dataset_dir
            metrics['objective_fitness'] = evaluator.get_objective_fitness(OBJECTIVE) / 100
            data.append(metrics)
        
        # Convert the data to a pandas DataFrame
        cur__df = pandas.DataFrame(data)
        cur__df.rename(columns={
            'dataset': 'Dataset',
            'objective_fitness': 'Objective Fitness',
        }, inplace=True)
        
        # Melt the DataFrame for Seaborn
        df_melted = cur__df.melt(id_vars='Dataset', 
                            value_vars=['Objective Fitness'],
                            var_name='Metric', 
                            value_name='Score')
    
        # Check if the file already exists if it does append to it
        if os.path.exists("./experiment_2/experiment_2.csv"):
            existing_df = pandas.read_csv("./experiment_2/experiment_2.csv")
            df_melted = pandas.concat([existing_df, df_melted], ignore_index=True)
    
        df_melted.to_csv("./experiment_2/experiment_2.csv", index=False)

def plot_data():
    # Load the data
    df_melted = pandas.read_csv("./experiment_2/experiment_2.csv")
    
    # rename the dataset values 
    df_melted['Dataset'] = df_melted['Dataset'].replace({
        '2019': '*2019',
        '2017': '*2017',
        '2020-id': '*2020-id',
        "2020-pl": "*2020-pl",
        "Nasa": "*Nasa",
        "RTFP": "*RTP",
        "2012": "*2012",
    })
        
    
    # Sort the DataFrame by 'Dataset' and 'Metric'
    df_melted.sort_values(by=['Dataset', 'Metric'], inplace=True)
    
    # Create a color palette
    #colors = cycle(px.colors.qualitative.Pastel2)
    fig = go.Figure()
    color = "lightgrey"
    for metric in ['Objective Fitness']:
        metric_df = df_melted[df_melted['Metric'] == metric]

        # Compute IQR per Dataset
        iqr_df = (
            metric_df.groupby('Dataset')['Score']
            .agg(lambda x: x.quantile(1) - x.quantile(0))
            .reset_index(name='IQR')
        )

        # Sort datasets by IQR
        sorted_datasets = iqr_df.sort_values('IQR')['Dataset'].tolist()

        # Filter color if needed
        color = color  # replace with your actual color logic if needed

        fig.add_trace(go.Box(
            x=metric_df['Dataset'],
            y=metric_df['Score'],
            name=metric,
            boxpoints='outliers',
            fillcolor=color,
            line={'width': 1, 'color': 'black'}
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
    fig.update_yaxes(range=[0.6, 1], dtick=0.1)
    
    # save the plot
    fig.write_image("./experiment_2/experiment_2.pdf")
    

if __name__ == "__main__":
    # raise NotImplementedError("This experiment is not implemented yet")
    # generate_data()
    
    plot_data()

    