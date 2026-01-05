import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import os

from model import DQN
from agent import Agent
from Connect4Env import Connect4
from CurriculumManager import CurriculumManager

def plot_rewards(scores, avg_scores, win_rates, filename_prefix="plots/rewards"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Rewards plot
    ax1.plot(scores, label="Episode Reward", alpha=0.3, linewidth=0.5)
    ax1.plot(avg_scores, label="50-Episode Average", color="red", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Reward Curve")
    ax1.legend()
    ax1.grid(True)
    
    # Win rate plot
    ax2.plot(win_rates, label="Win Rate (last 100 games)", color="green", linewidth=2)
    ax2.axhline(y=0.8, color='r', linestyle='--', label='Target (80%)')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Rate")
    ax2.set_title("Win Rate Over Time")
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}.png", dpi=150)
    plt.close()

def train(trial, stage_cfg):
    scores = []
    avg_scores = []
    results = []
    win_rates = []
    iteration = 0

    stage = stage_cfg["stage"]
    mode = stage_cfg["mode"]
    min_episodes = stage_cfg["episodes"]
    saveas = stage_cfg["saveas"]

    if mode == "selfplay":
        model_path = f"models/{trial}/{stage_cfg['model_path']}"
        past_agent = DQN(input_dim=2*6*7, fc1_dim=100, fc2_dim=100, n_action=7)
        past_agent.load_state_dict(torch.load(model_path))

        env = Connect4(mode=mode, agent=past_agent)
    else:
        env = Connect4(mode=mode)

    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    agent = Agent(input_dim=2*6*7, n_action=7)

    if mode == "selfplay":
        model_path = f"models/{trial}/{stage_cfg['model_path']}"
        print(f"Loading previous model from {model_path} into training agent...")
        agent.load_model(model_path)
        agent.eps = 1.0

    print(f"\nRunning Stage {stage}...")
    best_win_rate = 0
    while True:
        step = 0
        score = 0
        done = False
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs, env.get_non_empty_column())
            obs_, reward, done = env.step(action)
            score += reward
            agent.store_memory(obs, action, reward, done, obs_)
            loss = agent.learn()
            obs = obs_
            step += 1
            
        scores.append(score)

        # Track wins
        if reward == 1:
            results.append(1)
        elif reward == 0.5:  # tie
            results.append(0.5)
        else:
            results.append(0)

        avg_score = np.mean(scores[-50:])
        avg_scores.append(avg_score)
        
        # Calculate win rate over last 100 games
        window_size = min(100, len(results))
        recent_win_rate = np.mean(results[-window_size:])
        win_rates.append(recent_win_rate)

        if (iteration + 1) % 100 == 0:
            plot_rewards(scores, avg_scores, win_rates, 
                        f"figs/{trial}/rewards_{stage}")
            agent.plot_loss(f"figs/{trial}/loss_{stage}.png")
            
            # Calculate win rate over last 500 games for evaluation
            eval_window = min(500, len(results))
            eval_win_rate = np.mean(results[-eval_window:])
            
            print(f"Stage: {stage} | Episode: {iteration+1:5d} | "
                  f"Avg Score: {avg_score:6.3f} | "
                  f"Win Rate (100): {recent_win_rate:.3f} | "
                  f"Win Rate (500): {eval_win_rate:.3f} | "
                  f"Epsilon: {agent.eps:.4f} | "
                  f"Loss: {loss:.4f}")
            
            if eval_win_rate > best_win_rate:
                best_win_rate = eval_win_rate
                agent.save_model(f"models/{trial}/best_{saveas}")
        
        # Check if we've met the criteria
        if iteration + 1 >= min_episodes:
            # Need at least 1000 games for reliable win rate
            if iteration >= 1000:
                eval_window = 1000
                long_term_win_rate = np.mean(results[-eval_window:])
                
                if long_term_win_rate >= 0.75:  # 75% win rate threshold
                    print(f"\n✓ Stage {stage} completed! Win rate: {long_term_win_rate:.3f}")
                    break
            
        iteration += 1

        # Safety limit
        if iteration >= min_episodes * 3:
            print(f"\nReached safety limit. Final win rate: {np.mean(results[-500:]):.3f}")
            break

    agent.save_model(f"models/{trial}/{saveas}")
    
    # Save training statistics
    stats = {
        'final_win_rate': float(np.mean(results[-500:])),
        'best_win_rate': float(best_win_rate),
        'total_episodes': iteration + 1,
        'final_epsilon': float(agent.eps)
    }
    
    with open(f"models/{trial}/stats_{stage}.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Finished Stage {stage}. Model saved!")
    return stats

if __name__ == "__main__":
    trial = 1

    for dirname in ["figs", "models"]:
        os.makedirs(dirname, exist_ok=True)
        os.makedirs(f"{dirname}/{trial}", exist_ok=True)

    print(f"Trial: {trial}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    manager = CurriculumManager("curriculum_config.json")

    all_stats = {}
    for stage_cfg in manager.config:
        stats = train(trial, stage_cfg)
        all_stats[f"stage_{stage_cfg['stage']}"] = stats
    
    # Save overall statistics
    with open(f"models/{trial}/all_stats.json", 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print("Training Complete!")
    for stage_name, stats in all_stats.items():
        print(f"{stage_name}: Win Rate = {stats['final_win_rate']:.3f}, "
              f"Episodes = {stats['total_episodes']}")