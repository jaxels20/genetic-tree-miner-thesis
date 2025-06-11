# GTM.py

import argparse
from src.FileLoader import FileLoader
from src.Discovery import Discovery
from src.utils import load_hyperparameters_from_csv

def main(log_path: str, output_path: str, max_generations: int, time_limit: int, stagnation_limit: int):
    print(f"Loading log from: {log_path}")
    # Load the event log
    try:
        el = FileLoader.load_eventlog(log_path)
        print("Event log loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load event log: {e}")
    
    
    # Load hyperparameters
    try:
        hyperparameters = load_hyperparameters_from_csv('best_parameters.csv')
        hyperparameters['time_limit'] = time_limit
        hyperparameters['max_generations'] = max_generations
        hyperparameters['stagnation_limit'] = stagnation_limit
        
    except Exception as e:
        raise RuntimeError(f"Failed to load hyperparameters: {e}")
    

    # Run discovery algorithm
    try:
        print("Starting process discovery...")
        pn, pt = Discovery.genetic_algorithm(
            el, 
            **hyperparameters)
        print("Process discovery completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to discover process model: {e}")
    
    # Save the discovered model
    try:
        print("Saving the discovered model...")
        if output_path.endswith('.pnml'):
            pn.to_pnml(output_path)
        elif output_path.endswith('.pdf'):
            pn.visualize(output_path.removesuffix('.pdf'))
        else:
            raise ValueError("Output path must end with .pnml or .pdf")
    except Exception as e:
        raise RuntimeError(f"Failed to save the model: {e}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTM Process Discovery CLI")
    parser.add_argument("--log_path", required=True, help="Path to the input event log (.xes)")
    parser.add_argument("--output_path", required=True, help="Path to save the output either as .pnml or .pdf")

    parser.add_argument("--max_generations", type=int, default=None, help="Maximum number of generations for the genetic algorithm")
    parser.add_argument("--time_limit", type=int, default=None, help="Time limit for the genetic algorithm in seconds")
    parser.add_argument("--stagnation_limit", type=int, default=None, help="Stagnation limit for the genetic algorithm")

    args = parser.parse_args()

    main(args.log_path, args.output_path, args.max_generations, args.time_limit, args.stagnation_limit)