# Reborn and Refined: An Enhanced Genetic Algorithm for Process Discovery

This repository contains the code and data for the paper **_Reborn and Refined: An Enhanced Genetic Algorithm for Process Discovery_**. It implements a novel process discovery algorithm using a genetic approach and provides tools to reproduce the results and figures from the paper.

## 📂 Project Structure

   ```
├── requirements.txt          # Python dependencies required to run the project
├── GTM.py                    # The entrypoint for discovery
│
├── data/                     # Intermediate and processed data
│   ├── table_1/
│   └── ...
│
├── figures/                  # Final figures and tables for publication
│   ├── table_1.tex
│   └── ...
│   
├── real_life_datasets/       # Real-world event logs in XES format
│   ├── 2013-cp.xes
│   └── ...
│
├── produce_data/             # Scripts to generate raw data for experiments
│   ├── generate_figure_5a.py
│   └── ...
│  
└── produce_figures/          # Scripts to create figures and tables from data
       ├── generate_table_1.py
       └── ...
   ```


## 📖 Description

This repository accompanies our research on process discovery, introducing an enhanced genetic algorithm to generate accurate and interpretable process models from event logs. The approach is benchmarked on multiple real-life datasets and compared to existing methods.

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/jaxels20/genetic-tree-miner.git
   cd genetic-tree-miner
   ```
2. Install Dependencies
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Build and install FastTokenBasedReplay (needs a C++17 compiler; see [its README](src/FastTokenBasedReplay/README.md))
   ```bash
   pip install ./src/FastTokenBasedReplay
   ```
   
## ▶️ Usage
  ```bash
    python3 GTM.py --log_path logs/2013-cp.xes --output_path output.pdf --max_generations 5
  ```

## 📊 Datasets
The repository includes several real-life event logs from the 4TU Centre for Research Data. These are located in the event_logs/ folder and are in .xes format. However, please note that due to size limitations, only a subset of the event logs are included here, but they can all be downloaded [HERE](https://www.tf-pm.org/resources/logs) and put into the event log folder.

## 🧪 Reproducibility
Each script in produce_figures/ and produce_data/ generates a specific result from the paper, eg.
  ```bash
    # Generate Figure 5a
    python3 produce_figures/generate_figure_5a.py
    
    # Generate Table 2
    python3 produce_figures/generate_table_2.py
  ```

## 📜 License
This project is licensed under the terms of the MIT License. See LICENSE for more information.
## 📚 Citation
  ```
    @misc{

    }
  ```
