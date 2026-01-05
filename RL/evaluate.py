import torch

from model import DQN
from Connect4Env import Connect4

if __name__ == "__main__":
    model_path = "models/1/final_model.pth"
    agent = DQN(input_dim=2*6*7, fc1_dim=100, fc2_dim=100, n_action=7)
    agent.load_state_dict(torch.load(model_path))

    env = Connect4(agent=agent)

    env.play()