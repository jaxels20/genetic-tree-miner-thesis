
import pandas as pd
import os
import pickle
import plotly.graph_objects as go


INPUT_DIR = "./data/figure_5c/" 

OUTPUT_FILE = "./figures/figure_9.pdf"


def plot_data(df):
    # speeds are in seconds per generation
    speeds = {
        '2012': 11, 
        '2013-cp': 0.1, 
        '2013-i': 1.1, 
        '2013-op': 0.1, 
        '2017': 90, 
        '2019': 70, 
        '2020-dd': 0.2, 
        '2020-id': 1.5, 
        '2020-pl': 6.5, 
        '2020-ptc': 0.3, 
        '2020-rfp':5, 
        'RTF': 1.6, 
        'Sepsis': 0.6,
        }
    df['Speed'] = df['Dataset'].map(speeds)
    df['Time'] = df['Generation'] * df['Speed']
    
    # convert Time to minutes
    df['Time'] = df['Time'] / 60.0
    
    # Plotting the data
    colorblind_colors = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
    "#117733", "#332288", "#88CCEE", "#44AA99",
    "#661100", "#6699CC"
    ]

    marker_symbols = [
        "circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down",
        "triangle-left", "triangle-right", "star", "hexagram", "hourglass", "arrow", "bowtie",
    ]

    # Step 1: Compute max fitness per dataset
    dataset_order = (
        df.groupby("Dataset")["Fitness"]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    step = 10  # how often to place a marker
    fig = go.Figure()
    for i, dataset in enumerate(dataset_order):
        dataset_df = df[df["Dataset"] == dataset].reset_index(drop=True)
        
        # Main line without markers
        fig.add_trace(go.Scatter(
            x=dataset_df["Time"],
            y=dataset_df["Fitness"],
            mode='lines',
            showlegend=True,
            name=f"{dataset}",
            line=dict(color=colorblind_colors[i])
        ))
        

    # Step 3: Update layout
    fig.update_layout(
        title=None,
        xaxis_title="Time (Minutes)",
        yaxis_title="Objective Fitness",
        font=dict(family='Times New Roman', size=20),
        margin=dict(l=0, r=0, t=0, b=120),
        template='simple_white',
        legend=dict(
            font=dict(size=14, family="Time New Roman"),
            orientation="h",
            yanchor="bottom",
            #entrywidth=55,
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    fig.update_yaxes(
        range=[45, 100],
    )
    
    fig.update_xaxes(
        range=[0,15]
    )
    
    
    
    fig.write_image(OUTPUT_FILE, format='pdf')
    
    
def load_data():
    subfolders = [f.path for f in os.scandir(INPUT_DIR) if f.is_dir()]
    data = []
    
    for subfolder in subfolders:
        pkl_files = [f for f in os.listdir(subfolder + "/monitors") if f.endswith('.pkl')]
        
        for pkl_file in pkl_files:
            with open(os.path.join(subfolder, "monitors", pkl_file), 'rb') as f:
                dataset_name, method_name, result_dict = pickle.load(f)
                
            generations = list(result_dict.keys())
            fitness_values = list(result_dict.values())
            
            #generations = generations[::5]
            #fitness_values = fitness_values[::5]
            
            for i in range(len(generations)):
                data.append(
                    {
                        "Generation": generations[i],
                        "Fitness": fitness_values[i],
                        "Method": method_name,
                        "Dataset": dataset_name
                    }
                )
            
    # create a dataframe
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = load_data()
    plot_data(df)



