import gymnasium as gym
import torch
import numpy as np
from dqn import DQN # Or whatever your student class is named

# 1. Setup Device and Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("LunarLander-v3", render_mode="human") # human mode to watch it!

# 2. Load your Trained Student Model
# Ensure the dimensions match your trained BC model
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n
student_net = DQN(n_observations, n_actions).to(device) 

student_net.load_state_dict(torch.load("best_model.pth"))
student_net.eval() # Set to evaluation mode (turns off dropout/batchnorm)

# 3. Test Loop
num_test_episodes = 5

for episode in range(num_test_episodes):
    state, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Convert state to tensor for the model
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            # Get the action with the highest probability
            logits = student_net(state_tensor)
            action = torch.argmax(logits, dim=1).item()
        
        # Step the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        
        state = next_state
        total_reward += reward
        done = terminated or truncated
        
    print(f"Episode {episode + 1} Total Reward: {total_reward:.2f}")

env.close()