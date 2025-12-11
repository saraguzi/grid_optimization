from typing import Literal
import time
import os
import warnings

from Callbacks import HyperParamCallback, RewardLoggingCallback

from fleetrl.fleet_env.fleet_environment import FleetEnv
from fleetrl.benchmarking.benchmark import Benchmark
from fleetrl.benchmarking.uncontrolled_charging import Uncontrolled
from fleetrl.benchmarking.distributed_charging import DistributedCharging
from fleetrl.benchmarking.night_charging import NightCharging

from fleetrl.agent_eval.evaluation import Evaluation
from fleetrl.agent_eval.basic_evaluation import BasicEvaluation

from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


if __name__ == "__main__":
    input_data_path: str = "inputs"
    time_now = int(time.time())
    run_name: str = f"PPO_agent_{time_now}"
    n_train_steps = 168 #hours
    n_eval_steps = 720 #hours
    n_eval_episodes = 1
    n_evs = 1
    n_envs = 1
    time_steps_per_hour = 4
    use_case: str = "custom"
    custom_schedule_name = "1_lkw.csv"
    scenario: Literal["arb", "tariff"] = "arb"
    gen_new_schedule = False
    gen_new_test_schedule = False
    end_cutoff = 330

    ev_charger_power = 130  # kW
    grid_connection_limit = 500  # kW
    battery_size = 600  # kWh

    norm_obs_in_env = False
    vec_norm_obs = True
    vec_norm_rew = True

    total_steps = 100000
    saving_interval = total_steps // 10

    # Environment configuration
    env_config = {
        "data_path": input_data_path,
        "schedule_name": custom_schedule_name,
        "building_name": "load_lmd.csv",
        "pv_name": None,
        "use_case": use_case,
        "include_building": True,
        "include_pv": True,
        "include_price": True,
        "price_lookahead": 8,
        "bl_pv_lookahead": 4,
        "time_steps_per_hour": time_steps_per_hour,
        "time_picker": "random",
        "end_cutoff": end_cutoff,
        "deg_emp": False,
        "ignore_price_reward": False,
        "ignore_invalid_penalty": False,
        "ignore_overcharging_penalty": False,
        "ignore_overloading_penalty": False,
        "episode_length": n_train_steps,
        "normalize_in_env": norm_obs_in_env,
        "verbose": 0,
        "aux": True,
        "log_data": False,
        "calculate_degradation": True,
        "target_soc": 0.85,
        "seed": 42,
        "max_batt_cap_in_all_use_cases": 600,
        "init_battery_cap": 600,
        "init_soh": 1.0,
        "min_laxity": 1.75,
        "obc_max_power":  250,
        "real_time": False,
        "gen_schedule": gen_new_schedule,
        "gen_start_date": "2021-01-01 00:00",
        "gen_end_date": "2021-12-31 23:59:59",
        "gen_name": "my_custom_schedule.csv",
        "gen_n_evs": 1,
        "custom_ev_charger_power_in_kw": ev_charger_power,
        "custom_ev_battery_size_in_kwh": battery_size,
        "custom_grid_connection_in_kw": grid_connection_limit,

        # custom schedule timing settings, mean and standard deviation
        "custom_weekday_departure_time_mean": 7,
        "custom_weekday_departure_time_std": 1,
        "custom_weekday_return_time_mean": 19,
        "custom_weekday_return_time_std": 1,
        "custom_weekend_departure_time_mean": 9,
        "custom_weekend_departure_time_std": 1.5,
        "custom_weekend_return_time_mean": 17,
        "custom_weekend_return_time_std": 1.5,
        "custom_earliest_hour_of_departure": 3,
        "custom_latest_hour_of_departure": 11,
        "custom_earliest_hour_of_return": 12,
        "custom_latest_hour_of_return": 23,
        # custom distance settings
        "custom_weekday_distance_mean": 300,
        "custom_weekday_distance_std": 25,
        "custom_weekend_distance_mean": 150,
        "custom_weekend_distance_std": 25,
        "custom_minimum_distance": 20,
        "custom_max_distance": 400,
        # custom consumption data for vehicle
        "custom_consumption_mean": 1.3,
        "custom_consumption_std": 0.167463672468669,
        "custom_minimum_consumption": 0.3,
        "custom_maximum_consumption": 2.5,
        "custom_maximum_consumption_per_trip": 500
    }

    if scenario == "tariff":
        env_config["spot_markup"] = 10
        env_config["spot_mul"] = 1.5
        env_config["feed_in_ded"] = 0.25
        env_config["price_name"] = "spot_2021_new.csv"
        env_config["tariff_name"] = "fixed_feed_in.csv"
    else:
        env_config["spot_markup"] = 0
        env_config["spot_mul"] = 1
        env_config["feed_in_ded"] = 0
        env_config["price_name"] = "spot_2021_new.csv"
        env_config["tariff_name"] = "spot_2021_new_tariff.csv"

    env_kwargs = {"env_config": env_config}

    # Vector environments
    train_vec_env = make_vec_env(FleetEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs, seed=env_config["seed"])
    train_norm_vec_env = VecNormalize(train_vec_env, norm_obs=vec_norm_obs, norm_reward=vec_norm_rew, training=True)

    env_config["time_picker"] = "eval"
    eval_vec_env = make_vec_env(FleetEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs={"env_config": env_config}, seed=env_config["seed"] + 1)
    eval_norm_vec_env = VecNormalize(eval_vec_env, norm_obs=vec_norm_obs, norm_reward=vec_norm_rew, training=False)

    env_config["time_picker"] = "static"
    env_config["log_data"] = True
    test_vec_env = make_vec_env(FleetEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs={"env_config": env_config}, seed=env_config["seed"] + 2)
    test_norm_vec_env = VecNormalize(test_vec_env, norm_obs=vec_norm_obs, norm_reward=vec_norm_rew, training=False)

    # Callbacks
    eval_callback = EvalCallback(eval_norm_vec_env, warn=True, verbose=1, deterministic=True, eval_freq=max(10000 // n_envs, 1), n_eval_episodes=n_eval_episodes, render=False)
    progress_bar = ProgressBarCallback()
    hyperparameter_callback = HyperParamCallback()
    reward_callback = RewardLoggingCallback()

    # PPO model
    model = PPO(policy="MlpPolicy",
                env=train_norm_vec_env,
                verbose=1,
                tensorboard_log="./rl_agents/trained_agents/tb_log",
                n_steps = 2048)

    trained_agents_dir = f"./rl_agents/trained_agents/ppo_{int(time.time())}"
    os.makedirs(trained_agents_dir, exist_ok=True)

    # Training
    for i in range(0, int(total_steps / saving_interval)):
        print(f"Iteration {i}")
        model.learn(total_timesteps=saving_interval, reset_num_timesteps=False,
                    tb_log_name=f"{run_name}",
                    callback=[eval_callback, hyperparameter_callback, progress_bar, reward_callback])

        model.save(f"{trained_agents_dir}/{saving_interval * i}")
        stats_path = os.path.join(trained_agents_dir, f"vec_normalize-{saving_interval*i}.pkl")
        train_norm_vec_env.save(stats_path)

    # Save
    final_model_path = os.path.join(trained_agents_dir, "ppo_final.zip")
    model.save(final_model_path)
    stats_path = os.path.join(trained_agents_dir, "vec_normalize_final.pkl")
    train_norm_vec_env.save(stats_path)

    # Evaluation
    eval_agent: Evaluation = BasicEvaluation(n_steps=n_eval_steps, n_evs=n_evs, n_episodes=n_eval_episodes, n_envs=1)
    rl_log = eval_agent.evaluate_agent(env_kwargs={"env_config": env_config}, norm_stats_path=stats_path, model_path=final_model_path, seed=env_config["seed"])

    uc: Benchmark = Uncontrolled(n_steps=n_eval_steps, n_evs=n_evs, n_episodes=n_eval_episodes, n_envs=1, time_steps_per_hour=time_steps_per_hour)
    uc_log = uc.run_benchmark(env_kwargs={"env_config": env_config}, use_case=use_case, seed=env_config["seed"])

    dist: Benchmark = DistributedCharging(n_steps=n_eval_steps, n_evs=n_evs, n_episodes=n_eval_episodes, n_envs=1, time_steps_per_hour=time_steps_per_hour)
    dist_log = dist.run_benchmark(env_kwargs={"env_config": env_config}, use_case=use_case, seed=env_config["seed"])

    night: Benchmark = NightCharging(n_steps=n_eval_steps, n_evs=n_evs, n_episodes=n_eval_episodes, n_envs=1, time_steps_per_hour=time_steps_per_hour)
    night_log = night.run_benchmark(env_kwargs={"env_config": env_config}, use_case=use_case, seed=env_config["seed"])

    # Plotting
    uc.plot_benchmark(uc_log)
    dist.plot_benchmark(dist_log)
    night.plot_benchmark(night_log)

    eval_agent.compare(rl_log=rl_log, benchmark_log=uc_log, benchmark_name="Uncontrolled charging")
    eval_agent.compare(rl_log=rl_log, benchmark_log=dist_log, benchmark_name="Distributed charging")
    eval_agent.compare(rl_log=rl_log, benchmark_log=night_log, benchmark_name="Night charging")

    eval_agent.plot_soh(rl_log=rl_log, benchmark_log=uc_log, benchmark_name="Uncontrolled charging")
    eval_agent.plot_soh(rl_log=rl_log, benchmark_log=dist_log, benchmark_name="Distributed charging")
    eval_agent.plot_soh(rl_log=rl_log, benchmark_log=night_log, benchmark_name="Night charging")

    eval_agent.plot_violations(rl_log=rl_log)

    eval_agent.plot_action_dist(rl_log=rl_log, benchmark_log=uc_log, benchmark_name="Uncontrolled charging")
    eval_agent.plot_action_dist(rl_log=rl_log, benchmark_log=dist_log, benchmark_name="Distributed charging")
    eval_agent.plot_action_dist(rl_log=rl_log, benchmark_log=night_log, benchmark_name="Night charging")

    eval_agent.plot_cost(rl_log=rl_log, benchmark_log=uc_log, benchmark_name="Uncontrolled charging")
    eval_agent.plot_cost(rl_log=rl_log, benchmark_log=dist_log, benchmark_name="Distributed charging")
    eval_agent.plot_cost(rl_log=rl_log, benchmark_log=night_log, benchmark_name="Night charging")

    eval_agent.plot_cumulative_cost(rl_log, uc_log, dist_log, night_log)

    eval_agent.plot_training_rewards(reward_callback)