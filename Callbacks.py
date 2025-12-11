from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class HyperParamCallback(BaseCallback):
    """Saves hyperparameters and metrics at start of training, logging to tensorboard"""
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

class RewardLoggingCallback(BaseCallback):
    """
    Logs episodic rewards during training
    """
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.current_reward += reward

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.logger.record("train/ep_reward", self.current_reward)
            self.current_reward = 0.0

        return True