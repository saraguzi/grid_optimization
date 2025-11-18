import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG

from ev2gym.rl_agent.reward import V2G_profitmax
from ev2gym.rl_agent.state import V2G_profit_max

import utils
from pathlib import Path

# Config
config_file = "config.yaml"
reward_function = V2G_profitmax
state_function = V2G_profit_max

env = gym.make('EV2Gym-v1',
                config_file=config_file,
                reward_function=reward_function,
                state_function=state_function,
                save_replay=False,
                save_plots=True
)

simulation_length = env.unwrapped.simulation_length

# Dirs
history_dir = Path("history")
history_dir.mkdir(exist_ok=True)
history_files = []
stats_dir = Path("stats")
stats_dir.mkdir(exist_ok=True)

# Initialize the RL agent
models = [PPO("MlpPolicy", env), SAC("MlpPolicy", env), DDPG("MlpPolicy", env)]

for model in models:

    # Train the agent
    model.learn(total_timesteps=10000,
                progress_bar=True)

    # Evaluate the agent
    env = model.get_env()
    obs = env.reset()
    stats = []

    user_satisfaction_history = []
    energy_charged_history = []
    energy_discharged_history = []

    for i in range(simulation_length):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # TO DO history

        if done:
            stats.append(info)

    # Save stats
    if stats:
        model_name = model.__class__.__name__
        utils.save_stats_csv(stats, model_name)
        utils.save_stats_json(stats, model_name)

    # TO DO Save history


# Plots
csv_files = list(stats_dir.glob("stats_*.csv"))

utils.plot_total_reward(csv_files)
utils.plot_total_battery_degradation(csv_files)
utils.plot_total_profits(csv_files)

# TO DO Plot history