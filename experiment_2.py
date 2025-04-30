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

if __name__ == "__main__":
    # convert the hyper parameters to a normalize
    hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)
    
    dataset_dirs = os.listdir(DATASET_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{DATASET_DIR}{x}")]
    overall_df = pandas.DataFrame()
    for dataset_dir in dataset_dirs:
        if dataset_dir not in ["2013-cp", "2013-i", "2013-op"]:
            continue
        
        # Load the event log
        eventlog = EventLog.load_xes(f"{DATASET_DIR}{dataset_dir}/{dataset_dir}.xes")

        data = []
        for i in range(NUM_DATA_POINTS):
            discovered_net = Discovery.genetic_algorithm(
                eventlog,
                time_limit=60*5,
                stagnation_limit=50,
                **hyper_parameters
            )
            
            evaluator = SingleEvaluator(
                discovered_net,
                eventlog
            )
            
            # Get the evaluation metrics
            metrics = evaluator.get_evaluation_metrics(OBJECTIVE)
            metrics['dataset'] = dataset_dir
            metrics['objective_fitness'] = metrics['objective_fitness'] / 100
            data.append(metrics)
        
        # Convert the data to a pandas DataFrame
        cur__df = pandas.DataFrame(data)
        # concatenate the curr ent DataFrame with the overall DataFrame
        overall_df = pandas.concat([overall_df, cur__df], ignore_index=True)
    
     # Rename the columns
    overall_df.rename(columns={
        'log_fitness': 'Replay Fitness',
        'objective_fitness': 'Objective Fitness',
        'dataset': 'Dataset',
        'precision': 'Precision',
        'simplicity': 'Simplicity',
        'generalization': 'Generalization',
    }, inplace=True)
        
    # Melt the DataFrame for Seaborn
    df_melted = overall_df.melt(id_vars='Dataset', 
                        value_vars=['Replay Fitness', 'Objective Fitness'],
                        var_name='Metric', 
                        value_name='Score')
    df_melted.to_csv("./experiment_2/experiment_2.csv", index=False)

    fig = go.Figure()
    colors = cycle(px.colors.qualitative.Pastel2)
    
   # Add one boxplot trace for each metric
    for metric in ['Replay Fitness', 'Objective Fitness']:
        metric_df = df_melted[df_melted['Metric'] == metric]
        color = next(colors)
        fig.add_trace(go.Box(
            x=metric_df['Dataset'],
            y=metric_df['Score'],
            name=metric,
            boxpoints='outliers',  # show all points
            fillcolor=color,
            line={'width': 1, 'color': 'black'}
        ))

    # Layout adjustments
    fig.update_layout(
        boxmode='group',  # group boxes of same x-axis value
        font=dict(family='Arial', size=14),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            xanchor='center',
            y=1.05,        # slightly above the top of the plot
            x=0.5
        ),
        xaxis_title='Dataset',
        yaxis_title='Score',
        margin=dict(l=60, r=30, t=50, b=60),
        template='simple_white',
        height=500,
        width=900
    )

    # Constrain y-axis
    fig.update_yaxes(range=[0.8, 1])

    # save the plot
    fig.write_image("./experiment_2/experiment_2.pdf")