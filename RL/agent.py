import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import DQN

class Agent:
    def __init__(self, input_dim, n_action, lr=0.001, batch_size=64, eps=1.0, min_eps=0.05, gamma=0.99, 
                 fc1_dim=100, fc2_dim=100, fc3_dim=100, mem_size=50000, eps_dec=0.00001, 
                 target_update_freq=500):
        self.lr = lr
        self.batch_size = batch_size
        self.eps = eps
        self.eps_dec = eps_dec
        self.min_eps = min_eps
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.mem_size = mem_size
        self.mem_counter = 0
        self.learn_step_counter = 0
        self.action_space = [i for i in range(n_action)]

        # Main Q-network
        self.Q_eval = DQN(input_dim, fc1_dim, fc2_dim, n_action)
        
        # Target Q-network
        self.Q_target = DQN(input_dim, fc1_dim, fc2_dim, n_action)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_target.eval()  # Set to evaluation mode

        self.optimizer = torch.optim.Adam(self.Q_eval.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

        self.state_memory = np.zeros((self.mem_size, input_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, input_dim), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

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
            return 0.0
        
        self.optimizer.zero_grad()

        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(device=self.Q_eval.device)
        next_state_batch = torch.tensor(self.next_state_memory[batch]).to(device=self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(device=self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(device=self.Q_eval.device)
        action_batch = self.action_memory[batch]

        # Get current Q values
        q_curr = self.Q_eval(state_batch)[batch_index, action_batch]
        
        # Get next Q values from TARGET network (key improvement)
        with torch.no_grad():
            q_next = torch.max(self.Q_target(next_state_batch), dim=1)[0]
            q_next[terminal_batch] = 0.0
            q_target = reward_batch + self.gamma * q_next

        loss = self.criterion(q_curr, q_target)
        self.loss_history.append(loss.item())
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.learn_step_counter += 1

        # Update target network periodically
        if self.learn_step_counter % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
            print(f"Target network updated at step {self.learn_step_counter}")

        # Decay epsilon
        self.eps = self.eps - self.eps_dec if self.eps > self.min_eps else self.min_eps

        return loss.item()

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