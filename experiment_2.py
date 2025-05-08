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

TEST_DATASETS = ['2019', '2013-op', '2020-dd', '2020-ptc', "2020-rfp"]

def generate_data():
    # convert the hyper parameters to a normalize
    hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)
    
    dataset_dirs = os.listdir(DATASET_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{DATASET_DIR}{x}")]
    overall_df = pandas.DataFrame()
    for dataset_dir in dataset_dirs:
        if dataset_dir != TEST_DATASETS[0]:
            continue
        
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
            metrics['log_fitness'] = evaluator.get_replay_fitness()['log_fitness']
            metrics['dataset'] = dataset_dir
            metrics['objective_fitness'] = evaluator.get_objective_fitness(OBJECTIVE) / 100
            metrics['precision'] = evaluator.get_precision()
            metrics['f1_score'] = evaluator.get_f1_score(metrics['precision'], metrics['log_fitness'])
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
        'f1_score': 'F1 Score',
    }, inplace=True)
        
    # Melt the DataFrame for Seaborn
    df_melted = overall_df.melt(id_vars='Dataset', 
                        value_vars=['Replay Fitness', 'Objective Fitness', "Precision", "F1 Score"],
                        var_name='Metric', 
                        value_name='Score')
    
    # Check if the file already exists if it does append to it
    if os.path.exists("./experiment_2/experiment_2.csv"):
        # Load the existing data
        existing_df = pandas.read_csv("./experiment_2/experiment_2.csv")
        # Concatenate the new data with the existing data
        df_melted = pandas.concat([existing_df, df_melted], ignore_index=True)
    
    
    df_melted.to_csv("./experiment_2/experiment_2.csv", index=False)

def plot_data():
    # Load the data
    df_melted = pandas.read_csv("./experiment_2/experiment_2.csv")
    
    # Sort the DataFrame by 'Dataset' and 'Metric'
    df_melted.sort_values(by=['Dataset', 'Metric'], inplace=True)
    
    # Create a color palette
    colors = cycle(px.colors.qualitative.Pastel2)
    
    # Create a boxplot
    fig = go.Figure()
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
    # set the y axisto 0 to 1
    fig.update_yaxes(range=[0.6, 1], dtick=0.1)
    
    # save the plot
    fig.write_image("./experiment_2/experiment_2.pdf")
    

if __name__ == "__main__":
    raise NotImplementedError("This experiment is not implemented yet")
    generate_data()
    #plot_data()

    