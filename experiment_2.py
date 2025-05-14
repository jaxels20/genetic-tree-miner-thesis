import pandas
import os
import time
from src.Discovery import Discovery
from src.EventLog import EventLog 
from src.Evaluator import SingleEvaluator
import plotly.graph_objects as go
import plotly.express as px
from experiment_1 import load_hyperparameters_from_csv

BEST_PARAMS = "./best_parameters.csv"
DATASET_DIR = "./real_life_datasets/"
NUM_DATA_POINTS = 5
TIME_LIMIT = 60*5
STAGNATION_LIMIT = 50
PERCENTAGE_OF_LOG = 0.05
OBJECTIVE = {
    "simplicity": 10,
    "refined_simplicity": 10,
    "ftr_fitness": 50,
    "ftr_precision": 30
}

def generate_data():
    # convert the hyper parameters to a normalize
    hyper_parameters = load_hyperparameters_from_csv(BEST_PARAMS)
    dataset_dirs = os.listdir(DATASET_DIR)
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{DATASET_DIR}{x}")]

    for dataset_dir in dataset_dirs:
        eventlog = EventLog.load_xes(f"{DATASET_DIR}{dataset_dir}/{dataset_dir}.xes")

        data = []
        for i in range(NUM_DATA_POINTS):
            print(f"Running discovery on dataset: {dataset_dir} iteration: {i}")
            start = time.time()
            discovered_net = Discovery.genetic_algorithm(
                eventlog,
                time_limit=TIME_LIMIT,
                stagnation_limit=STAGNATION_LIMIT,
                percentage_of_log=PERCENTAGE_OF_LOG,
                **hyper_parameters,
            )
            time_taken = time.time() - start
            
            # Export the discovered net to a file
            os.makedirs(f"./geneticminer/pdfs", exist_ok=True)
            discovered_net.visualize(f"./geneticminer/pdfs/{dataset_dir}_{i}")
            os.makedirs(f"./geneticminer/pnmls", exist_ok=True)
            discovered_net.to_pnml(f"./geneticminer/pnmls/{dataset_dir}_{i}")
            
            evaluator = SingleEvaluator(
                discovered_net,
                eventlog
            )
            
            # Get the evaluation metrics
            fitness = evaluator.get_replay_fitness()['log_fitness']
            precision = evaluator.get_precision()
            
            metrics = {}
            metrics['dataset'] = dataset_dir
            metrics['miner'] = "GM"
            metrics['model'] = i
            metrics['log_fitness'] = fitness
            metrics['precision'] = precision
            metrics['f1_score'] = evaluator.get_f1_score(precision, fitness)
            metrics['objective_fitness'] = evaluator.get_objective_fitness(OBJECTIVE)
            metrics['generalization'] = evaluator.get_generalization()
            metrics['simplicity'] = evaluator.get_simplicity()
            metrics['time'] = time_taken
            data.append(metrics)
        
        # Convert the data to a pandas DataFrame
        cur_df = pandas.DataFrame(data)        
        cur_df_copy = cur_df.copy()
        if os.path.exists(f"./experiment_1/results_genetic_{NUM_DATA_POINTS}_runs.csv"):
            read_df = pandas.read_csv(f"./experiment_1/results_genetic_{NUM_DATA_POINTS}_runs.csv")
            cur_df_copy = pandas.concat([read_df, cur_df_copy], ignore_index=True)
        
        cur_df_copy.to_csv(f"./experiment_1/results_genetic_{NUM_DATA_POINTS}_runs.csv", index=False)
        
        # Melt the DataFrame for Seaborn
        df_melted = cur_df.melt(id_vars='dataset', var_name='Metric', value_name='Score')
        df_melted = df_melted[df_melted['Metric'] == 'objective_fitness']

        if os.path.exists("./experiment_2/experiment_2.csv"):
            old = pandas.read_csv("./experiment_2/experiment_2.csv")
            df_melted = pandas.concat([old, df_melted], ignore_index=True)  
        
        df_melted.to_csv("./experiment_2/experiment_2.csv", index=False)

def plot_data():
    # Load the data
    df_melted = pandas.read_csv("./experiment_2/experiment_2.csv")
    # Remove all rows with Nasa as the dataset
    df_melted.rename(columns={'dataset': 'Dataset'}, inplace=True)
    df_melted = df_melted[~df_melted['Dataset'].str.contains("Nasa")]
    
    # rename the dataset values
    df_melted.rename(columns={'dataset': 'Dataset'}, inplace=True)
    df_melted['Dataset'] = df_melted['Dataset'].replace({
        '2019': '*2019',
        '2017': '*2017',
        '2020-id': '*2020-id',
        "2020-pl": "*2020-pl",
        "RTF": "*RTF",
        "2012": "*2012",
    })
    
    
    # Sort the DataFrame by 'Dataset' and 'Metric'
    df_melted.sort_values(by=['Dataset', 'Metric'], inplace=True)
        
    # Create a color palette
    #colors = cycle(px.colors.qualitative.Pastel2)
    fig = go.Figure()
    color = "lightgrey"
    for metric in ['objective_fitness']:
        metric_df = df_melted[df_melted['Metric'] == metric]

        # Compute IQR per Dataset
        iqr_df = (
            metric_df.groupby('Dataset')['Score']
            .agg(lambda x: x.quantile(1) - x.quantile(0))
            .reset_index(name='IQR')
        )

        # Sort datasets by IQR
        sorted_datasets = iqr_df.sort_values('IQR')['Dataset'].tolist()

        fig.add_trace(go.Scatter(
            x=metric_df['Dataset'],
            y=metric_df['Score'],
            mode='markers',
            name=metric,
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
    fig.write_image("./experiment_2/experiment_2.pdf")
    

if __name__ == "__main__":
    # generate_data()
    plot_data()

    