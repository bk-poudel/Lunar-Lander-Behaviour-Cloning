import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. THE ARCHITECTURE (Must match your training script exactly)
class BCLunarLander(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(BCLunarLander, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def test_model(model_path="best_model.pth", num_episodes=5):
    # 2. SETUP ENVIRONMENT
    # render_mode="human" is what opens the GUI window
    env = gym.make("LunarLander-v3", render_mode="human")
    
    n_actions = 4
    n_observations = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. INITIALIZE AND LOAD MODEL
    model = BCLunarLander(n_observations, n_actions).to(device)
    
    try:
        # Loading the dictionary you saved in your trainer
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Successfully loaded model! (Validation Accuracy was: {checkpoint.get('val_accuracy', 'N/A')})")
    except FileNotFoundError:
        print(f"Error: Could not find {model_path}. Did you finish training yet?")
        return

    model.eval() # Set to evaluation mode (stops training math)

    # 4. THE GUI LOOP
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Prepare state tensor (MLP expects batch dimension)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad(): # No gradient calculation needed for testing
                logits = model(state_tensor)
                # Pick the action with the highest raw score (logit)
                action = torch.argmax(logits, dim=1).item()
            
            # Apply the action to the GUI environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            state = next_state
            total_reward += reward
            done = terminated or truncated
            
        print(f"Episode {episode + 1} Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test_model()