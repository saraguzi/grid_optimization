import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import ast

# Helpers

def ensure_dir(path_str):
    path = Path(path_str)
    path.mkdir(exist_ok=True)
    return path

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(x) for x in obj]
    else:
        return obj

def flatten(row):
    if isinstance(row, dict):
        new_row = {}
        for k, v in row.items():
            if isinstance(v, (np.integer, np.floating)):
                new_row[k] = v.item()
            elif isinstance(v, np.ndarray):
                new_row[k] = [convert(x) for x in v]
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    new_row[f"{k}.{subk}"] = convert(subv)
            elif isinstance(v, list):
                new_row[k] = [convert(x) for x in v]
            else:
                new_row[k] = v
        return new_row
    elif isinstance(row, (list, np.ndarray)):
        return [convert(x) for x in row]
    else:
        return row

# Saving

def save_stats_csv(stats, model_name, stats_dir="stats"):
    path = ensure_dir(stats_dir)
    timestamp = get_timestamp()
    filename = f"stats_{model_name}_{timestamp}.csv"
    filepath = path / filename

    flat_stats = [flatten(row) for row in stats]
    df = pd.DataFrame(flat_stats)
    df.to_csv(filepath, index=False)
    return filepath

def save_stats_json(stats, model_name, stats_dir="stats"):
    path = ensure_dir(stats_dir)
    timestamp = get_timestamp()
    filename = f"stats_{model_name}_{timestamp}.json"
    filepath = path / filename

    clean_stats = [convert(row) for row in stats]

    with open(filepath, 'w') as f:
        json.dump(clean_stats, f, indent=4)

    return filepath

def save_history_csv(user_satisfaction, energy_charged, energy_discharged, model_name, history_dir="history"):
    # TO DO
    pass

def extract_info(info, key, default=np.nan):
    if isinstance(info, list):
        for item in info:
            if key in item:
                return convert(item[key])
    elif isinstance(info, dict):
        return convert(info.get(key, default))
    return default

# Load
def load_csv_stats(csv_file):
    df_raw = pd.read_csv(csv_file)
    df_dicts = df_raw.iloc[:, 0].apply(ast.literal_eval)
    df = pd.json_normalize(df_dicts)
    return df

# Plots
def plot_total_reward(csv_files):
    agents = []
    values = []
    for file in csv_files:
        df = load_csv_stats(file)
        agent_name = Path(file).stem.split("_")[1]
        if 'total_reward' in df.columns:
            agents.append(agent_name)
            values.append(df['total_reward'].sum())
    plt.figure(figsize=(8,5))
    plt.bar(agents, values, color=['skyblue','salmon','lightgreen'])
    plt.title("Total Reward per Agent")
    plt.ylabel("Total Reward")
    plt.show()

def plot_total_battery_degradation(csv_files):
    agents = []
    values = []
    for file in csv_files:
        df = load_csv_stats(file)
        agent_name = Path(file).stem.split("_")[1]
        if 'battery_degradation' in df.columns:
            agents.append(agent_name)
            values.append(df['battery_degradation'].sum())
    plt.figure(figsize=(8,5))
    plt.bar(agents, values, color=['skyblue','salmon','lightgreen'])
    plt.title("Battery Degradation per Agent")
    plt.ylabel("Battery Degradation")
    plt.show()

def plot_total_profits(csv_files):
    agents = []
    values = []
    for file in csv_files:
        df = load_csv_stats(file)
        agent_name = Path(file).stem.split("_")[1]
        if 'total_profits' in df.columns:
            agents.append(agent_name)
            values.append(df['total_profits'].sum())
    plt.figure(figsize=(8,5))
    plt.bar(agents, values, color=['skyblue','salmon','lightgreen'])
    plt.title("Total Profits per Agent")
    plt.ylabel("Total Profits")
    plt.show()

def plot_history():
    # TO DO
    pass