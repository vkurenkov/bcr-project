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
from tensorboardX import SummaryWriter
from functools import partial

import torch.multiprocessing as mp

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

    def forward(self, obs: torch.tensor):
        raise NotImplementedError()

    def sample_action(self, obs: torch.tensor):
        raise NotImplementedError()

    def action_entropy(self, obs: torch.tensor) -> float:
        """
        Return entropy on the distribution of actions.
        """
        raise NotImplementedError()

    def init_from_parameters(self, parameters):
        cur_ind = 0
        for param in self.parameters():
            size = param.numel()
            param.data = parameters[cur_ind:cur_ind+size].view(param.data.size())
            cur_ind += size

class MujocoPolicy(Policy):
    def __init__(self, obs_size, act_size):
        super(MujocoPolicy, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.action_mean = nn.Linear(256, act_size)
        self.action_std  = nn.Sequential(
            nn.Linear(256, act_size)
        )

        self.obs_size = obs_size
        self.act_size = act_size

        normc_initializer(1.0, self.state_encoder)
        normc_initializer(0.01, self.action_mean)
        normc_initializer(0.01, self.action_std)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        encoded_state = self.state_encoder.forward(obs)

        mean     = self.action_mean.forward(encoded_state)
        log_std  = self.action_std.forward(encoded_state)    
        return mean, torch.exp(log_std)

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean, std = self.forward(obs)

        return distrib.Normal(mean, std).sample()

    def action_entropy(self, obs: torch.tensor) -> float:
        with torch.no_grad():
            mean, std = self.forward(obs)

        entropies = distrib.Normal(mean, torch.diagflat(std)).entropy()
        return torch.sum(torch.diag(entropies))

class ClassicControlPolicy(Policy):
    def __init__(self, obs_size, act_size):
        super(ClassicControlPolicy, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(obs_size, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh()
        )
        self.action_mean = nn.Linear(5, act_size)
        self.action_std  = nn.Sequential(
            nn.Linear(5, act_size),
            nn.ReLU()
        )

        self.obs_size = obs_size
        self.act_size = act_size

        normc_initializer(self.state_encoder)
        normc_initializer(self.action_mean)
        normc_initializer(self.action_std)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        encoded_state = self.state_encoder.forward(obs)

        mean = self.action_mean.forward(encoded_state)
        std  = self.action_std.forward(encoded_state)    
        return mean, std

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean, std = self.forward(obs)

        return distrib.Normal(mean, std).sample()

    def action_entropy(self, obs: torch.tensor) -> float:
        with torch.no_grad():
            mean, std = self.forward(obs)
        
        return distrib.Normal(mean, std).entropy().item()


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
    def _transform_train_rew(self, rew, action, prev_obs, obs):
        entropy = self.policy.action_entropy(self._transform_obs(prev_obs))
        return rew + 0.01 * entropy

class ESPopulation:
    def __init__(self, num_agents: int, num_trials:int, lr: float,
        initial_agent: ESAgent,agent_class, env_name: str, weights_std: float,
        seed: int, num_parallel: int):
        self.num_agents  = num_agents
        self.num_trials  = num_trials
        self.env_name    = env_name

        self.agent       = initial_agent
        self.agent_class = agent_class
        self.seed        = seed
        self.lr          = lr    

        self.centroid            = deepcopy(initial_agent.policy)
        self.weights_std         = weights_std
        self.num_parameters      = count_parameters(self.centroid)
        self.pertrubations_distr = distrib.Normal(torch.zeros(self.num_parameters), self.weights_std)
        
        self.num_parallel = num_parallel

    def step(self) -> Tuple[List[float], List[float]]:
        """
        Optimizes the entire population with antithetic sampling.
        Returns training population rewards and timesteps for the current step.
        """
        perturbs = self._sample_pertrubations(self.num_agents)
        perturbs_rew                  = []
        perturbs_timesteps            = []
        centroid_parameters           = unroll_parameters(self.centroid.parameters())

        report_rew = []
        
        # rollout_params = [self.num_trials, self.env_name, self.seed]
        # res = []
        # for j in range(0,self.num_agents,self.num_parallel): 
        #     jobs = []
        #     pipe_list = []
        #     for ind in range(0, self.num_parallel):
        #         r_e,s_e = mp.Pipe(False)
        #         p = mp.Process(target = _step,args = (perturbs[j],deepcopy(self.agent_class),
        #         deepcopy(self.centroid),deepcopy(rollout_params),s_e ))
        #         jobs.append(p)
        #         pipe_list.append(r_e)
        #         p.start()
            
        #     for proc in jobs:
        #         proc.join()
        #     result_list = [p.recv() for p in pipe_list]
        #     res.extend(result_list)


        # perturbs_rew,perturbs_timesteps,pt2,report_rew,rr2 = list(zip(*res))
        # perturbs_timesteps = list(perturbs_timesteps)
        # report_rew = list(report_rew)
        # perturbs_timesteps.extend(list(pt2))
        # report_rew.extend(list(rr2))

        for ind in range(0, self.num_agents):
            perturb      = perturbs[ind]

            # Initialize agent with perturbed parameters
            self.agent.policy.init_from_parameters(centroid_parameters + perturb)
            reward, timesteps = self.agent.train_rollout(self.num_trials, self.env_name, self.seed)
            
            # Initialize agent with anti perturbed parameters
            self.agent.policy.init_from_parameters(centroid_parameters - perturb)
            reward_anti, timesteps_anti = self.agent.train_rollout(self.num_trials, self.env_name, self.seed)

            perturbs_rew.append(reward - reward_anti)

            perturbs_timesteps.append(timesteps)
            perturbs_timesteps.append(timesteps_anti)
            report_rew.append(reward)
            report_rew.append(reward_anti)

        # Transform rewards as in Salimans et al. (2017)
        transformed_rews = compute_centered_ranks(np.array(perturbs_rew))

        # GRADIENT ASCENT
        perturbs = np.stack(perturbs)
        for ind in range(0, self.num_agents):
            grad = torch.tensor(transformed_rews[ind] * perturbs[ind])
            centroid_parameters += grad * (self.lr) / (2 * self.num_agents * self.weights_std**2)

        # Update the centroid
        self.centroid.init_from_parameters(centroid_parameters)
        print("Gradient norm: {}".format(torch.norm(grad, p=2)))

        return report_rew, perturbs_timesteps
    
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

    def _sample_pertrubations(self, num_samples: int) -> [torch.tensor]:
        pertrubations           = []
        for _ in range(num_samples):
            sampled_pertrubation = self.pertrubations_distr.sample()
            # To be optimized
            pertrubations.append(sampled_pertrubation)
        
        return pertrubations

def _step(perturb, agent_class,centroid,rollout_params : list,send_end):
        centroid_parameters  = unroll_parameters(centroid.parameters())
        local_agent = agent_class(centroid)

        # Initialize agent with perturbed parameters
        local_agent.policy.init_from_parameters(centroid_parameters + perturb)
        reward, timesteps = local_agent.train_rollout(*rollout_params)
        
        # Initialize agent with anti perturbed parameters
        local_agent.policy.init_from_parameters(centroid_parameters - perturb)
        reward_anti, timesteps_anti = local_agent.train_rollout(*rollout_params)
        
        perturbs_rew = reward - reward_anti
        send_end.send([perturbs_rew,timesteps,timesteps_anti,reward,reward_anti])
        
seed       = 0
env_name   = "Hopper-v2"
fix_random_seeds(seed)
writer     = SummaryWriter()


env        = gym.make(env_name)
policy     = MujocoPolicy(len(env.observation_space.high), len(env.action_space.high))
agent      = ESAgent(policy)
population = ESPopulation(num_agents=40, num_trials=5, lr=0.01,
                initial_agent=agent,agent_class=ESAgent,
                env_name=env_name, weights_std=0.02, seed=seed,num_parallel=3)

for cur_iter in range(10000):
    print("\nIteration #{}".format(cur_iter))
    print("--------------------------")
    train_rews,  train_steps = population.step()
    test_reward, test_steps  = population.test()

    print("Train population mean reward: {} (+/-{})".format(np.mean(train_rews), np.std(train_rews)))
    print("Train population mean steps: {} (+/-{})".format(np.mean(train_steps), np.std(train_steps)))
    print("Test centroid reward: {}".format(test_reward))
    print("Test centroid steps: {}".format(test_steps))

    writer.add_scalar("Training reward", np.mean(train_rews), cur_iter)
    writer.add_scalar("Training steps",  np.mean(train_steps), cur_iter)
    writer.add_scalar("Test reward", test_reward, cur_iter)
    writer.add_scalar("Test steps",  test_steps,  cur_iter)