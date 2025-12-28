import gymnasium as gym
import torch
import numpy as np

# 1. Setup Environment
env = gym.make("LunarLander-v3", render_mode="human") # human mode to watch the live flight

# 2. Re-create the Model Architecture
# Important: This must be identical to the one you used for training
from BC_lunar_lander import BCLunarLander # Import your class
n_observations = 8
n_actions = 4
model = BCLunarLander(n_observations, n_actions)

# 3. Load the "Best" Weights
checkpoint = torch.load("best_model.pth", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval() # CRITICAL: Sets model to evaluation mode

print(f"Loaded model with Validation Accuracy: {checkpoint['val_accuracy']:.4f}")

# 4. Test Loop
num_episodes = 5
for ep in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Convert state to tensor and add batch dimension (1, 8)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(state_t)
            # Use argmax to pick the action with the highest probability
            action = torch.argmax(logits, dim=1).item()
        
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
    print(f"Episode {ep+1} finished with Reward: {total_reward:.2f}")

env.close()