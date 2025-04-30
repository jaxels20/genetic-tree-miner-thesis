import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.EventLog import EventLog
from src.PetriNet import PetriNet, Marking
from src.Evaluator import SingleEvaluator
import src.FastTokenBasedReplay as FastTokenBasedReplay
from itertools import cycle
import plotly.express as px

import plotly.graph_objects as go

def generate_petri_net(lenght = 40):
    pn = PetriNet()
    pn.empty()
    pn.add_place("start")
    pn.add_place("end")
    pn.add_place("p1")
    pn.add_place("p2")
    pn.add_place("p3")
    
    
    pn.add_transition("A")
    pn.add_transition("B")
    pn.add_transition("tau_0")
    pn.add_transition("tau_1")
    
    
    pn.add_arc("start", "tau_0")
    pn.add_arc("tau_0", "p1")
    pn.add_arc("p1", "B")
    pn.add_arc("B", "p2")
    
    pn.add_arc("p2", "tau_1")
    pn.add_arc("tau_1", "p3")
    pn.add_arc("p3", "A")
    pn.add_arc("A", "end")
    

    pn.add_place("p4")
    pn.add_arc("tau_0", "p4")
    prev_place = "p4"
    for i in range(10, lenght+10):
        pn.add_place(f"p{i}")
        pn.add_transition(f"tau_{i+1}")
        pn.add_arc(prev_place, f"tau_{i+1}")
        pn.add_arc(f"tau_{i+1}", f"p{i}")
        prev_place = f"p{i}"
    pn.add_arc(f"p{lenght+9}", "tau_1")


    inial_marking = Marking({"start": 1})

    pn.set_initial_marking(inial_marking)

    final_marking = Marking({"end": 1})
    pn.set_final_marking(final_marking)

    return pn


if __name__ == "__main__":
    data = {"size": [], "fitness": []}
    for i in range(1, 50, 5):
        pn = generate_petri_net(i)
        el = EventLog.from_trace_list(["A"])

        eval = SingleEvaluator(pn, el)
        
        data["size"].append(i)
        data["fitness"].append(eval.get_replay_fitness()['log_fitness'])
        

    ftr_fitness =  FastTokenBasedReplay.calculate_fitness(el.to_fast_token_based_replay(), pn.to_fast_token_based_replay(), False, False)

    fig = go.Figure()
    colors = cycle(px.colors.qualitative.Pastel2)
    fig.add_trace(go.Scatter(
        x=data["size"],
        y=data["fitness"],
        mode='lines+markers',
        name="Replay Fitness",
        
        line=dict(color="black", width=1),
        marker=dict(symbol='x', size=8, line=dict(width=1, color='black'))
    ))
    
    fig.update_layout(
        xaxis_title="k",
        yaxis_title="Replay Fitness",
        template="simple_white",
        font=dict(size=12),
        width=600,
    )
    
    
    # Save the figure as pdf
    fig.write_image("./fitness_pm4py_point/fitness.pdf", format="pdf")

    








