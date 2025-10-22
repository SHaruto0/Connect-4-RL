import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import DQN

class Agent:
    def __init__(self, input_dim, n_action, lr=0.001, batch_size=64, eps=1.0, min_eps=0.01, gamma=0.99, fc1_dim=256, fc2_dim=256, fc3_dim=256,
                 mem_size=5000, eps_dec=0.0001):
        self.lr = lr
        self.batch_size = batch_size
        self.eps = eps
        self.eps_dec = eps_dec
        self.min_eps = min_eps
        self.gamma = gamma

        self.mem_size = mem_size
        self.mem_counter = 0
        self.action_space = [i for i in range(n_action)]

        self.Q_eval = DQN(input_dim, fc1_dim, fc2_dim, fc3_dim, n_action)

        self.optimizer = torch.optim.Adam(self.Q_eval.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.state_memory = np.zeros((self.mem_size, input_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, input_dim), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        self.loss_history = []

    def store_memory(self, state, action, reward, terminal, next_state):
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal

        self.mem_counter += 1
    
    def choose_action(self, obs, non_full_column):
        if np.random.random() > self.eps:
            with torch.no_grad():
                obs = torch.tensor([obs], dtype=torch.float32, device=self.Q_eval.device)
                q_val = self.Q_eval(obs).squeeze(0)

                mask = torch.tensor(non_full_column == 0, device=self.Q_eval.device)
                q_val[mask] = -float("inf")
                action = torch.argmax(q_val).item()
        else:
            action = np.random.choice(np.where(non_full_column == 1)[0].tolist())

        return action


    def learn(self):
        if self.mem_counter < self.batch_size:
            return
        
        self.optimizer.zero_grad()

        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(device=self.Q_eval.device)
        next_state_batch = torch.tensor(self.next_state_memory[batch]).to(device=self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(device=self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(device=self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_curr = self.Q_eval(state_batch)[batch_index,action_batch]
        q_pred = self.Q_eval(next_state_batch)
        q_pred[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_pred, dim=1)[0]

        loss = self.criterion(q_curr, q_target).to(self.Q_eval.device)
        self.loss_history.append(loss.item())
        loss.backward()
        self.optimizer.step()

        self.eps = self.eps - self.eps_dec if self.eps > self.min_eps else self.min_eps

    def plot_loss(self, file_path="figs/loss.png"):
        plt.plot(np.arange(len(self.loss_history)), self.loss_history)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Loss History")
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

    def save_model(self, model_path="model/model.pth"):
        torch.save(self.Q_eval.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path="model/model.pth"):
        self.Q_eval.load_state_dict(torch.load(model_path))