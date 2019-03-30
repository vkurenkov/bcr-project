import torch
import torch.nn as nn
import torch.distributions as distrib
import gym
import numpy as np

from copy import deepcopy
from utils import unroll_parameters, count_parameters
from utils import fill_policy_parameters, normc_initializer
from utils import compute_centered_ranks, fix_random_seeds
from typing import Tuple, List


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

    def forward(self, obs: torch.tensor):
        raise NotImplementedError()

    def sample_action(self, obs: torch.tensor):
        raise NotImplementedError()

    def init_from_parameters(self, parameters):
        cur_ind = 0
        for param in self.parameters():
            size = param.numel()
            param.data = parameters[cur_ind:cur_ind+size].view(param.data.size())
            cur_ind += size

class MujocoPolicy(Policy):
    def __init__(self, obs_size, act_size, act_std):
        super(MujocoPolicy, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.action_mean = nn.Linear(64, act_size)

        self.obs_size = obs_size
        self.act_size = act_size
        self.act_std  = act_std

        normc_initializer(self.state_encoder)
        normc_initializer(self.action_mean)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_mean.forward(self.state_encoder.forward(obs))

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean = self.forward(obs)

        return distrib.Normal(mean, self.act_std).sample()

class ClassicControlPolicy(Policy):
    def __init__(self, obs_size, act_size, act_std):
        super(ClassicControlPolicy, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(obs_size, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh()
        )
        self.action_mean = nn.Linear(5, act_size)

        self.obs_size = obs_size
        self.act_size = act_size
        self.act_std  = act_std

        normc_initializer(self.state_encoder)
        normc_initializer(self.action_mean)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_mean.forward(self.state_encoder.forward(obs))

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean = self.forward(obs)

        return distrib.Normal(mean, self.act_std).sample()


class ESAgent:
    def __init__(self, policy: Policy):
        self.policy = policy

    def _transform_obs(self, obs):
        return torch.tensor(obs, dtype=torch.float32).view(1, -1)

    def _transform_train_rew(self, rew, action, prev_obs, obs) -> float:
        return rew

    def _trajectory_rew(self, rewards) -> float:
        return np.sum(rewards)

    def gradient(self, num_trials, env_name: str, seed: int) -> torch.tensor:
        raise NotImplementedError()

    def test_rollout(self, num_trials: int, env_name: str, seed: int) -> Tuple[float, float]:
        """
        Returns mean reward and mean number of steps.
        """
        self.policy.eval()

        env = gym.make(env_name)
        env.seed(seed)

        rollout_rewards   = []
        rollout_timesteps = []
        for _ in range(num_trials):
            rewards = []

            done     = False
            obs      = env.reset()
            timestep = 0
            while not done:
                action = self.policy.sample_action(self._transform_obs(obs))
                obs, rew, done, _ = env.step(action)
                rewards.append(rew)
                timestep += 1

            rollout_rewards.append(self._trajectory_rew(rewards))
            rollout_timesteps.append(timestep)

        return np.mean(rollout_rewards), np.mean(rollout_timesteps)

    def train_rollout(self, num_trials: int, env_name: str, seed: int) -> Tuple[float, float]:
        """
        Returns mean training reward and mean number of steps.
        """
        self.policy.eval()

        env = gym.make(env_name)
        env.seed(seed)

        rollout_rewards   = []
        rollout_timesteps = []
        for _ in range(num_trials):
            rewards = []

            done     = False
            obs      = env.reset()
            timestep = 0
            while not done:
                action = self.policy.sample_action(self._transform_obs(obs))
                prev_obs = deepcopy(obs)
                obs, rew, done, _ = env.step(action)
                rewards.append(self._transform_train_rew(rew, action, prev_obs, obs))

                timestep += 1

            rollout_timesteps.append(timestep)
            rollout_rewards.append(self._trajectory_rew(rewards))

        return np.mean(rollout_rewards), np.mean(rollout_timesteps)
    
class ESMaxEntropyAgent(ESAgent):
    pass

class ESPopulation:
    def __init__(self, num_agents: int, num_trials:int, lr: float,
        initial_agent: ESAgent, env_name: str, weights_std: float, seed: int):
        self.num_agents  = num_agents
        self.num_trials  = num_trials
        self.env_name    = env_name

        self.agent       = initial_agent
        self.seed        = seed
        self.lr          = lr    

        self.centroid            = deepcopy(initial_agent.policy)
        self.weights_std         = weights_std
        self.num_parameters      = count_parameters(self.centroid)
        self.pertrubations_distr = distrib.Normal(torch.zeros(self.num_parameters), self.weights_std)

    def step(self) -> Tuple[List[float], List[float]]:
        """
        Optimizes the entire population.
        Returns training population rewards and timesteps for the current step.
        """
        perturbs            = self._sample_pertrubations(self.num_agents, antithetic=True)
        perturbs_rew        = []
        perturbs_timesteps  = []
        centroid_parameters = unroll_parameters(self.centroid.parameters())
        for perturb in perturbs:
            # Initialize agent with perturbed parameters
            self.agent.policy.init_from_parameters(centroid_parameters + perturb)
            # Do simulations
            reward, timesteps = self.agent.train_rollout(self.num_trials, self.env_name, self.seed)
            
            perturbs_rew.append(reward)
            perturbs_timesteps.append(timesteps)

        # Transform rewards as in Salimans et al. (2017)
        transformed_rews = compute_centered_ranks(np.array(perturbs_rew))

        # GRADIENT ASCENT
        perturbs = np.stack(perturbs)
        for ind in range(0, self.num_agents):
            grad = torch.tensor(transformed_rews[ind] * perturbs[ind])
            centroid_parameters += grad * (self.lr) / (self.num_agents * self.weights_std)

        # Update the centroid
        self.centroid.init_from_parameters(centroid_parameters)
        print("Gradient norm: {}".format(torch.norm(grad, p=2)))

        return perturbs_rew, perturbs_timesteps

    def test(self) -> Tuple[float, float]:
        """
        Tests the centroid.
        """
        self.agent.policy.init_from_parameters(unroll_parameters(self.centroid.parameters()))
        reward, timesteps = self.agent.test_rollout(1, self.env_name, self.seed)

        return reward, timesteps

    def save(self, path) -> None:
        """
        Saves the centroid agent.
        """
        with open(path, mode="wb") as f:
            torch.save(self.centroid, f)

    def _sample_pertrubations(self, num_samples: int, antithetic: bool=True) -> [torch.tensor]:
        pertrubations       = []
        for _ in range(num_samples):
            sampled_pertrubation = self.pertrubations_distr.sample()
            # To be optimized
            pertrubations.append(sampled_pertrubation)
            if antithetic:
                pertrubations.append(-sampled_pertrubation)
        
        return pertrubations


seed       = 0
env_name   = "Hopper-v2"
fix_random_seeds(seed)


env        = gym.make(env_name)
policy     = MujocoPolicy(len(env.observation_space.high), len(env.action_space.high), 0.01)
agent      = ESAgent(policy)
population = ESPopulation(num_agents=40, num_trials=5, lr=0.01,
                initial_agent=agent, env_name=env_name, weights_std=0.02, seed=seed)

for cur_iter in range(10000):
    print("\nIteration #{}".format(cur_iter))
    print("--------------------------")
    train_rews,  train_steps = population.step()
    test_reward, test_steps  = population.test()

    print("Train population mean reward: {} (+/-{})".format(np.mean(train_rews), np.std(train_rews)))
    print("Train population mean steps: {} (+/-{})".format(np.mean(train_steps), np.std(train_steps)))
    print("Test centroid reward: {}".format(test_reward))
    print("Test centroid steps: {}".format(test_steps))