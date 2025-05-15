from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max
import warnings
import gymnasium as gym
from stable_baselines3 import DDPG

config_file = "ev2gym/example_config_files/V2GProfitPlusLoads.yaml"
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize the environment
env = EV2Gym(config_file=config_file,
              save_replay=False,
              save_plots=True)

state, _ = env.reset()

# Optimal solution
# agent = V2GProfitMaxOracle(env, verbose=True)

# or heuristic
agent = ChargeAsFastAsPossible()

for t in range(env.simulation_length):
    # get action from the agent
    actions = agent.get_action(env)
    # takes action
    new_state, reward, done, truncated, stats = env.step(actions)


## Train an RL agent, using the StableBaselines3 library
reward_function = profit_maximization
state_function = V2G_profit_max

env = gym.make('EV2Gym-v1',
                config_file=config_file,
                reward_function=reward_function,
                state_function=state_function)

# Initialize the RL agent
# use Deep Deterministic Policy Gradient
model = DDPG("MlpPolicy", env)

# Train the agent
model.learn(total_timesteps=1000,
            progress_bar=True)

# Evaluate the agent
env = model.get_env()
obs = env.reset()
stats = []
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    if done:
        stats.append(info)