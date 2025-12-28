import gymnasium as gym
import torch
import os
import numpy as np
from BC_lunar_lander import BCLunarLander

env=gym.make("LunarLander-v3", render_mode="human")
num_episodes=100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_model_path=os.path.join("best_model.path")
