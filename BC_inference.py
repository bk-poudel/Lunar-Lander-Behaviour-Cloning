import gymnasium as gym
import torch
import os
import numpy as np
from itertools import count # Added this import
from BC_lunar_lander import BCLunarLander

# 1. Setup Environment
env = gym.make("LunarLander-v3", render_mode="human")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Correct File Path (Ensure this matches your saved file name!)
policy_model_path = "best_model.pth" 

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# 3. Initialize Model
policy_net = BCLunarLander(n_observations, n_actions).to(device)

# 4. LOAD LOGIC FIX
# Since your training script saved a dictionary, we must access "model_state_dict"
checkpoint = torch.load(policy_model_path, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    policy_net.load_state_dict(checkpoint["model_state_dict"])
else:
    policy_net.load_state_dict(checkpoint)

policy_net.eval() 

# 5. Testing Loop
num_episodes = 10
for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0
    
    for t in count(): # count() runs until 'break' is called
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            # Get action: find index of the max logit
            action_tensor = policy_net(state_tensor).max(1).indices.view(1, 1)
        
        # Step the environment
        next_state, reward, terminated, truncated, info = env.step(action_tensor.item())
        
        state = next_state
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode {episode+1} Reward: {total_reward:.2f}")
            break

env.close()