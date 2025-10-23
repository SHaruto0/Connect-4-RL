import numpy as np
import matplotlib.pyplot as plt
import torch
import json

from model import DQN
from agent import Agent
from Connect4Env import Connect4
from CurriculumManager import CurriculumManager

def plot_rewards(scores, avg_scores, filename="plots/rewards.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label="Reward", alpha=0.5)
    plt.plot(avg_scores, label="Moving Average", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train(trial, stage_cfg):
    scores = []
    avg_scores = []
    results = []
    iteration = 0

    stage = stage_cfg["stage"]
    mode = stage_cfg["mode"]
    min_episodes = stage_cfg["episodes"]
    saveas = stage_cfg["saveas"]

    if mode == "selfplay":
        model_path = stage_cfg["model_path"]
        past_agent = DQN(input_dim=2*6*7, fc1_dim=256, fc2_dim=256, n_action=7)
        past_agent.load_state_dict(torch.load(model_path))

        env = Connect4(mode=mode, agent=past_agent)
    else:
        env = Connect4(mode=mode)

    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    agent = Agent(input_dim=2*6*7, n_action=7)

    print(f"\nRunning Stage {stage}...")
    while True:
        step = 0
        score = 0
        done = False
        obs = env.reset()

        while not done:
            # print(obs)
            action = agent.choose_action(obs, env.get_non_empty_column())
            # print(action)
            obs_, reward, done = env.step(action)
            score += reward
            agent.store_memory(obs, action, reward, done, obs_)
            agent.learn()
            obs = obs_
            step += 1
        scores.append(score)

        if reward == 1:
            results.append(1)
        # elif reward == 0.5:
        #     results.append(0.5)
        else:
            results.append(0)

        avg_score = np.mean(scores[-50:])
        avg_scores.append(avg_score)

        if (iteration + 1) % 250 == 0:
            plot_rewards(scores, avg_scores, f"figs/rewards_{stage}_{trial}.png")
            agent.plot_loss(f"figs/loss_{stage}_trial_{trial}.png")
            
            lower_bound = max(iteration - 2500 + 1, 0)
            recent_results = results[lower_bound:iteration + 1]
            win_rate = np.sum(recent_results) / len(recent_results)
            
            print(f"Stage: {stage} \t Episode: {iteration+1} \t Average Score: {avg_score} \t Win Rate: {win_rate}")
        
        if iteration + 1 >= min_episodes:
            lower = max(0, iteration - 2499)
            window = results[lower:iteration+1]
            win_rate = np.sum(window) / len(window)
            if win_rate >= 0.8:
                break

        iteration += 1

    agent.save_model(saveas)
    print(f"Finished Stage {stage}. Saved model!!")

if __name__ == "__main__":
    trial = 2

    manager = CurriculumManager("curriculum_config.json")

    for stage_cfg in manager.config:
        train(trial, stage_cfg)