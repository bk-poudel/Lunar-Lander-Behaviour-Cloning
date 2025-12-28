import gymnasium as gym
import torch
import os
import numpy as np
from BC_lunar_lander import BCLunarLander

env=gym.make("LunarLander-v3", render_mode="human")
num_episodes=100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_model_path=os.path.join("best_model.path")
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n
policy_net = BCLunarLander(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load(policy_model_path))
