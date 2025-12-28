import gymnasium as gym
import torch
import numpy as np
import time
from BC_lunar_lander import BCLunarLander
# 1. Setup the Environment with GUI enabled
# render_mode="human" triggers the visual window
env = gym.make("LunarLander-v3", render_mode="human")

# 2. Setup Model (Ensure this matches your BCLunarLander class)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BCLunarLander(n_observations=8, n_actions=4).to(device)

# 3. Load your specific saved dictionary
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Starting GUI Test. Watch the popup window!")

for episode in range(5):
    state, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Standardize state for the MLP
        state_t = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(state_t)
            action = torch.argmax(logits, dim=1).item()
        
        # Step the environment
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # Optional: slow down the GUI if it's too fast to see
        # time.sleep(0.01) 

    print(f"Episode {episode+1} Reward: {total_reward:.2f}")

env.close()