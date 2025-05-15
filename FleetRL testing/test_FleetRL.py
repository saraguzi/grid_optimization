# Importing dependencies

from typing import Literal
import pandas as pd
import time

from fleetrl.fleet_env.fleet_environment import FleetEnv
from fleetrl.benchmarking.benchmark import Benchmark
from fleetrl.benchmarking.uncontrolled_charging import Uncontrolled
from fleetrl.benchmarking.distributed_charging import DistributedCharging
from fleetrl.benchmarking.night_charging import NightCharging

from fleetrl.agent_eval.evaluation import Evaluation
from fleetrl.agent_eval.basic_evaluation import BasicEvaluation

from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, BaseCallback
from stable_baselines3.common.logger import HParam

from pink import PinkActionNoise

import warnings
warnings.simplefilter(action='ignore')

if __name__ == "__main__":
    # Environment
    env = FleetEnv("./tutorial.json")
    env.reset()
    # Take an action in the environment
    # Actions can range from -1 to 1 - corresponding to the % of kW of the charger
    # Charging
    env.step([1])
    new_soc = 30 * 0.91 / 600 + 0.6012
    price_with_fees = 0.084
    energy_drawn_at_charger = 30
    cost = energy_drawn_at_charger * price_with_fees
    print(cost)
    # Discharging
    env.step([-1])
    new_soc = 0.647 - 30/600
    useful_energy = 30*0.91
    deduction = 0.25
    revenue = useful_energy * (0.062 * (1-deduction))
    print(revenue)
    for _ in range(10):
        env.step([10])
    # define fundamental parameters
    # data path to inputs folder with schedule csv, prices, load, pv, etc.
    input_data_path: str = "inputs"  # path to input folder
    time_now = int(time.time())
    run_name: str = f"Test_run_custom_{time_now}"  # Change this name or make it dynamic, e.g. with a timestamp
    n_train_steps = 48  # number of hours in a training episode
    n_eval_steps = 168  # number of hours in one evaluation episode
    n_eval_episodes = 1  # number of episodes for evaluation
    n_evs = 1  # number of evs
    n_envs = 2  # number of environment running in parallel (speeds up training, 1 env = 1 CPU)
    time_steps_per_hour = 4  # temporal resolution of the simulation (quarter-hourly)
    use_case: str = "custom"  # for file name - lmd=last mile delivery, by default can insert "lmd", "ct", "ut", "custom"
    custom_schedule_name = "1_lkw.csv"  # name for custom schedule if you have generated one. If you want to generate one this time, this field will be ignored
    scenario: Literal[
        "arb", "tariff"] = "arb"  # arbitrage or tariff. Arbitrage allows for bidirectional spot trading, no fees. Tariff models commercial tariff with grid fees, electricity tax, etc.
    gen_new_schedule = False  # generate a new schedule - refer to schedule generator documentation and adjust statistics in config.json
    gen_new_test_schedule = False  # generate a new schedule for agent testing
    end_cutoff = 330  # the dataset has 365 days. We want to only train on 1 month of data in this lab, so we remove the last 11 months

    # we will be simulating a truck
    ev_charger_power = 130  # kW
    grid_connection_limit = 500  # kW
    battery_size = 600  # kWh

    # training parameters
    norm_obs_in_env = False  # normalize observations within FleetRL (max, min normalization)
    vec_norm_obs = True  # normalize observations in SB3 (rolling normalization)
    vec_norm_rew = True  # normalize rewards in SB3 (rolling normalization)

    # Total steps should be sep to 1e6 or 5e6 for a full run. Check tensorboard for stagnating reward signal and stop training at some point to avoid overfit
    total_steps = int(5e3)  # total training time steps

    # Specifies how often you want to make an intermediate artifact. For a full run, I recommend every 50k - 100k steps, so you can backtrack for best model
    saving_interval = 5e2  # interval for saving the model

    # environment arguments - adjust settings if necessary
    # additional settings can be changed in the config files
    env_config = {"data_path": input_data_path,
                  # Specify file names: there is a naming convention for default files, otherwise, custom name is used
                  "schedule_name": (str(n_evs) + "_" + str(
                      use_case) + ".csv") if use_case != "custom" else custom_schedule_name,
                  "building_name": "load_" + str(use_case) + ".csv" if use_case != "custom" else "load_lmd.csv",
                  "pv_name": None,
                  # if separate file for PV inputs, specify here, otherwise, uses "PV" column in building_name
                  # Define use case
                  "use_case": use_case,
                  # Change observation space
                  "include_building": True,  # False removes building load from Observation
                  "include_pv": True,  # False removes PV from Observation
                  "include_price": True,  # False removes electricity prices from Observation
                  "price_lookahead": 8,  # Hours seen into the future price
                  "bl_pv_lookahead": 4,  # Hours seen into the future building load and pv
                  "time_steps_per_hour": 4,  # Time resolution
                  # Specify time picker: "eval", "static", or "random" are implemented
                  "time_picker": "random",  # Pick a random starting day in the schedule dataframe
                  "end_cutoff": end_cutoff,  # how many days to remove from the end of the dataset
                  # Pick degradation methodology: True sets empirical degradation from real measurements
                  "deg_emp": False,  # empirical degradation calculation
                  # Shape reward function
                  "ignore_price_reward": False,  # True sets price-related reward coefficient to 0
                  "ignore_invalid_penalty": False,  # True ignores penalties on invalid actions (charging an empty spot)
                  "ignore_overcharging_penalty": False,  # True ignores penalties on charging signals above target SOC
                  "ignore_overloading_penalty": False,  # True ignores grid connection overloading penalty
                  # Set episode length during training
                  "episode_length": n_train_steps,  # in hours
                  # Additional parameters
                  "normalize_in_env": norm_obs_in_env,  # Conduct normalization within FleetRL.
                  "verbose": 0,  # Print statements, can slow down FPS
                  "aux": True,  # Include auxiliary data (recommended). Check documentation for more information.
                  "log_data": False,  # Log data (Makes most sense for evaluation runs)
                  "calculate_degradation": True,  # Calculate SOH degradation (Can slow down FPS)
                  # Target SOC
                  "target_soc": 0.85,  # Signals that would charge above target SOC are clipped.
                  # settings regarding the generation of evs
                  # seed for random number generation
                  "seed": 42,  # Seed for RNG - can be set to None so always random (not recommended)
                  # if you are comparing cars with different bess sizes, use this to norm their reward function range
                  "max_batt_cap_in_all_use_cases": 600,
                  "init_battery_cap": 600,
                  # initial state of health of the battery
                  "init_soh": 1.0,
                  "min_laxity": 1.75,
                  "obc_max_power": 250,
                  "real_time": False,
                  # settings below if you want to generate new schedules
                  "gen_schedule": gen_new_schedule,  # boolean to generate a new schedule
                  "gen_start_date": "2021-01-01 00:00",  # if new schedule, start date
                  "gen_end_date": "2021-12-31 23:59:59",  # if new schedule, end date
                  "gen_name": "my_custom_schedule.csv",  # name of newly generated schedule
                  "gen_n_evs": 1,  # number of EVs in new schedule, per EV it takes ca. 10-20 min.
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
                  "custom_maximum_consumption_per_trip": 500,
                  # custom ev-related settings
                  "custom_ev_charger_power_in_kw": ev_charger_power,
                  "custom_ev_battery_size_in_kwh": battery_size,
                  "custom_grid_connection_in_kw": grid_connection_limit
                  }

    # commercial tariff scenario, fixed fee on spot price (+10 ct/kWh, and a 50% mark-up)
    # Feed-in tariff orientates after PV feed-in, with 25% deduction
    if scenario == "tariff":
        env_config["spot_markup"] = 10
        env_config["spot_mul"] = 1.5
        env_config["feed_in_ded"] = 0.25
        env_config["price_name"] = "spot_2021_new.csv"
        env_config["tariff_name"] = "fixed_feed_in.csv"

    # arbitrage scenario, up and down prices are spot price, no markups or taxes
    elif scenario == "arb":
        env_config["spot_markup"] = 0
        env_config["spot_mul"] = 1
        env_config["feed_in_ded"] = 0
        env_config["price_name"] = "spot_2021_new.csv"
        env_config["tariff_name"] = "spot_2021_new_tariff.csv"

    env_kwargs = {"env_config": env_config}

    train_vec_env = make_vec_env(FleetEnv,
                                 n_envs=n_envs,
                                 vec_env_cls=SubprocVecEnv,
                                 env_kwargs=env_kwargs,
                                 seed=env_config["seed"])

    train_norm_vec_env = VecNormalize(venv=train_vec_env,
                                      norm_obs=vec_norm_obs,
                                      norm_reward=vec_norm_rew,
                                      training=True,
                                      clip_reward=10.0)

    env_config["time_picker"] = "eval"

    if gen_new_schedule:
        env_config["gen_schedule"] = False
        env_config["schedule_name"] = env_config["gen_name"]

    env_kwargs = {"env_config": env_config}

    eval_vec_env = make_vec_env(FleetEnv,
                                n_envs=n_envs,
                                vec_env_cls=SubprocVecEnv,
                                env_kwargs=env_kwargs,
                                seed=env_config["seed"])

    eval_norm_vec_env = VecNormalize(venv=eval_vec_env,
                                     norm_obs=vec_norm_obs,
                                     norm_reward=vec_norm_rew,
                                     training=True,
                                     clip_reward=10.0)

    if gen_new_test_schedule:
        # generate an evaluation schedule
        test_sched_name = env_config["gen_name"]
        if not test_sched_name.endswith(".csv"):
            test_sched_name = test_sched_name + "_test" + ".csv"
        else:
            test_sched_name = test_sched_name.strip(".csv")
            test_sched_name = test_sched_name + "_test" + ".csv"

        env_config["gen_schedule"] = True
        env_config["gen_name"] = test_sched_name

        env_kwargs = {"env_config": env_config}

        test_vec_env = make_vec_env(FleetEnv,
                                    n_envs=1,
                                    vec_env_cls=SubprocVecEnv,
                                    env_kwargs=env_kwargs,
                                    seed=env_config["seed"])

        env_config["gen_schedule"] = False
        env_config["schedule_name"] = test_sched_name

        env_kwargs = {"env_config": env_config}

    test_vec_env = make_vec_env(FleetEnv,
                                n_envs=n_envs,
                                vec_env_cls=SubprocVecEnv,
                                env_kwargs=env_kwargs,
                                seed=env_config["seed"])

    test_norm_vec_env = VecNormalize(venv=test_vec_env,
                                     norm_obs=vec_norm_obs,
                                     norm_reward=vec_norm_rew,
                                     training=True,
                                     clip_reward=10.0)
    eval_callback = EvalCallback(eval_env=eval_norm_vec_env,
                                 warn=True,
                                 verbose=1,
                                 deterministic=True,
                                 eval_freq=max(10000 // n_envs, 1),
                                 n_eval_episodes=5,
                                 render=False,
                                 )


    class HyperParamCallback(BaseCallback):
        """
        Saves hyperparameters and metrics at start of training, logging to tensorboard
        """

        def _on_training_start(self) -> None:
            hparam_dict = {
                "algorithm": self.model.__class__.__name__,
                "learning rate": self.model.learning_rate,
                "gamma": self.model.gamma,
            }

            metric_dict = {
                "rollout/ep_len_mean": 0,
                "train/value_loss": 0.0,
            }

            self.logger.record(
                "hparams",
                HParam(hparam_dict, metric_dict),
                exclude=("stdout", "log", "json", "csv")
            )

        def _on_step(self) -> bool:
            return True


    progress_bar = ProgressBarCallback()

    hyperparameter_callback = HyperParamCallback()

    # model-related settings
    n_actions = train_norm_vec_env.action_space.shape[-1]
    param_noise = None
    noise_scale = 0.1
    seq_len = n_train_steps * time_steps_per_hour
    action_noise = PinkActionNoise(noise_scale, seq_len, n_actions)

    load_existing_agent = True
    if load_existing_agent:
        model = PPO.load(
            path="./rl_agents/trained_agents/LMD_arbitrage_1e6_steps_example_agent/PPO-fleet_LMD_2021_arbitrage_PPO_mul3.zip",
            env=train_norm_vec_env)


    # Agent evaluation
    # environment arguments for evaluation
    env_config["time_picker"] = "static"  # Pick a random starting day in the schedule dataframe
    env_config["log_data"] = True,  # Log data (Makes most sense for evaluation runs)

    env_kwargs = {"env_config": env_config}

    eval: Evaluation = BasicEvaluation(n_steps=n_eval_steps,
                                       n_evs=n_evs,
                                       n_episodes=n_eval_episodes,
                                       n_envs=1)

    if load_existing_agent:
        stats_path = "./rl_agents/trained_agents/LMD_arbitrage_1e6_steps_example_agent/vec_normalize-LMD_2021_arbitrage_PPO_mul3.pkl"
        model_path = "./rl_agents/trained_agents/LMD_arbitrage_1e6_steps_example_agent/PPO-fleet_LMD_2021_arbitrage_PPO_mul3.zip"

    # steps through a week of data and calls the agent's learnt strategy
    rl_log = eval.evaluate_agent(env_kwargs=env_kwargs, norm_stats_path=stats_path, model_path=model_path,
                                 seed=env_config["seed"])


    print(rl_log)

    # Benchmarks

    uncontrolled_charging: Benchmark = Uncontrolled(n_steps=n_eval_steps,
                                                    n_evs=n_evs,
                                                    n_episodes=n_eval_episodes,
                                                    n_envs=1,
                                                    time_steps_per_hour=time_steps_per_hour)

    uc_log = uncontrolled_charging.run_benchmark(env_kwargs=env_kwargs, use_case=use_case, seed=env_config["seed"])


    dist: Benchmark = DistributedCharging(n_steps=n_eval_steps, n_evs=n_evs, n_episodes=n_eval_episodes, n_envs=1,
                                          time_steps_per_hour=time_steps_per_hour)

    dist_log = dist.run_benchmark(env_kwargs=env_kwargs, use_case=use_case, seed=env_config["seed"])

    night: Benchmark = NightCharging(n_steps=n_eval_steps, n_evs=n_evs, n_episodes=n_eval_episodes, n_envs=1,
                                     time_steps_per_hour=time_steps_per_hour)

    night_log = night.run_benchmark(env_kwargs=env_kwargs, use_case=use_case, seed=env_config["seed"])

    uncontrolled_charging.plot_benchmark(uc_log)

    eval.compare(rl_log=rl_log, benchmark_log=uc_log)
    eval.plot_soh(rl_log=rl_log, benchmark_log=uc_log)
    eval.plot_soh(rl_log=rl_log, benchmark_log=dist_log)
    eval.plot_soh(rl_log=rl_log, benchmark_log=night_log)
    eval.plot_violations(rl_log=rl_log, benchmark_log=uc_log)
    eval.plot_action_dist(rl_log=rl_log, benchmark_log=uc_log)

    detailed_fig = eval.plot_detailed_actions(start_date="2021-01-02 19:00",
                                              end_date="2021-01-04 18:45",
                                              rl_log=rl_log,
                                              uc_log=uc_log,
                                              dist_log=dist_log,
                                              night_log=night_log)

    detailed_fig.show()

    print(rl_log)


    def get_from_obs(log: dict):

        obs = log["Observation"]
        act = log["Charging energy"]
        cf = log["Cashflow"]
        env_config = env_kwargs["env_config"]

        bl_pv_lookahead = env_config["bl_pv_lookahead"]
        pr_lookahead = env_config["price_lookahead"]

        length = len(log)

        # Check observer class to see how observation list is built up

        date = log["Time"]
        first = 0  # first entry has index 0
        last = n_evs - 1  # soc for each car
        if n_evs > 1:
            soc = [obs[i][first:last].mean() for i in range(length)]
        else:
            soc = [obs[i][first] for i in range(length)]

        first = n_evs
        last = n_evs * 2 - 1  # hours left at charger for each car
        if n_evs > 1:
            hours_left = [obs[i][first:last].mean() for i in range(length)]
        else:
            hours_left = [obs[i][first] for i in range(length)]

        first = n_evs * 2
        last = n_evs * 2 + pr_lookahead  # price lookahead gives price in hour, hour+1, etc.
        price = [obs[i][first] for i in range(length)]

        first = n_evs * 2 + pr_lookahead + 1
        last = n_evs * 2 + pr_lookahead * 2 + 1  # tariff paid when discharging, with lookahead
        tariff = [obs[i][first] for i in range(length)]

        first = n_evs * 2 + pr_lookahead * 2 + 2
        last = n_evs * 2 + pr_lookahead * 2 + 2 + bl_pv_lookahead  # building load lookahead
        building_load = [obs[i][first] for i in range(length)]

        first = n_evs * 2 + pr_lookahead * 2 + 2 + bl_pv_lookahead + 1
        last = n_evs * 2 + pr_lookahead * 2 + bl_pv_lookahead * 2 + 1  # pv has same lookahead as building
        pv = [obs[i][first] for i in range(length)]

        free_cap = [obs[i][-8] / obs[i][-9] for i in range(length)]  # free grid capacity in MW / total grid capacity

        time_steps_per_hour = env_config["time_steps_per_hour"]

        # act is charging energy in kWh, we want to display the currently drawn power in kW
        first = 0  # first entry has index 0
        last = n_evs - 1  # soc for each car
        if n_evs > 1:
            action = [act[i][first:last].sum() * time_steps_per_hour for i in range(length)]  # Going from kWh to kW
        else:
            action = [act[i][first] * time_steps_per_hour for i in range(length)]  # Going from kWh to kW

        df = pd.DataFrame({
            'Date': date,
            'SOC': soc,
            'Load': building_load,
            'PV': pv,
            'Price': price,
            'Action': action,
            'Free cap': free_cap,
            'CF': cf
        })

        return df

    get_from_obs(rl_log)

