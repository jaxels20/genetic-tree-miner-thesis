from src.utils import calculate_percentage_of_log
import pandas as pd
import plotly.graph_objects as go

data = [
    ["Sepsis", 846],
    ["2020-ptc", 202],
    ["2012", 4366],
    ["RTF", 231],
    ["2017", 15930],
    ["2020-rfp", 89],
    ["2020-dd", 99],
    ["2020-id", 753],
    ["2013-i", 1511],
    ["2020-pl", 1478],
    ["2013-op", 108],
    ["2019", 11028],
    ["2013-cp", 183],
]

df = pd.DataFrame(data, columns=["Dataset", "Number of Unique Traces"])

datasets_to_show_text = [
    "Sepsis",
    "2019",
    "2012",
    "2017",
    "2020-pl",
    "2013-cp"
]



df["Percentage of Log"] = df["Number of Unique Traces"].apply(
    lambda x: calculate_percentage_of_log(x))

text_df = df[df["Dataset"].isin(datasets_to_show_text)]

log_percentage_x = list(range(1, 20000, 100))
log_percentage_y = [calculate_percentage_of_log(x) for x in log_percentage_x]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=log_percentage_x,
    y=log_percentage_y,
    mode='lines',
    name='Percentage of Log',
    line=dict(color='black', width=2),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=df["Number of Unique Traces"],
    y=df["Percentage of Log"],
    mode='markers',
    name='Datasets',
    text=df["Dataset"],
    textposition='top center',
    marker=dict(size=10, color='red', symbol='circle'),
    textfont=dict(size=12, family='Times New Roman'),
    showlegend=False,
    )
)  
fig.add_trace(go.Scatter(
    x=text_df["Number of Unique Traces"],
    y=text_df["Percentage of Log"],
    mode='markers+text',
    name='Datasets',
    text=text_df["Dataset"],
    textposition='top center',
    marker=dict(size=10, color='red', symbol='circle'),
    textfont=dict(size=12, family='Times New Roman'),
    showlegend=False,
    )
)
# add percentage suffix to y-axis ticks
fig.update_layout(
    title="Percentage of Log vs Number of Unique Traces",
    xaxis_title="Number of Unique Traces",
    yaxis_title="Percentage of Log (%)",
    yaxis_tickformat='.0%',  # show 1 decimal + percentage
)
# Step 3: Update layout
fig.update_layout(
    title=None,
    font=dict(family='Times New Roman', size=20),
    margin=dict(l=0, r=0, t=0, b=120),
    template='simple_white',
)

fig.write_image("percentage_of_log.pdf")

        