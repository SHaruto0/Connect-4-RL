# Connect 4 Reinforcement Learning (DQN)

A Deep Q-Network (DQN) agent that learns to play Connect 4 through self-play and curriculum learning.  
A web interface allows users to play directly in their browser against the trained AI.

---

## Overview

This project implements an end-to-end Connect 4 reinforcement learning system.  
The agent is trained to master the game from scratch using a Deep Q-Network and progressively harder opponents through curriculum learning.  
The trained model is deployed via AWS SageMaker and served through a FastAPI backend, with a React frontend for browser-based gameplay.

---

## Features

- **Custom Connect 4 Environment**: Built from scratch with complete game logic
- **Deep Q-Network (DQN)**: Epsilon-greedy exploration with experience replay
- **Curriculum Learning**: Progressive training from random opponents to self-play
- **Model Checkpointing**: Saves intermediate models for evaluation and deployment
- **Web Interface**: React frontend with FastAPI backend for live gameplay
- **Cloud Deployment**: Model hosted on AWS SageMaker for scalable inference

---

## Project Structure

```
Connect4-RL/
│
├── RL/
│   ├── main.py                    # Training entry point
│   ├── agent.py                   # DQN agent logic
│   ├── model.py                   # Neural network architecture
│   ├── Connect4Env.py             # Connect 4 environment
│   ├── CurriculumManager.py       # Curriculum manager (staged training)
│   ├── curriculum_config.json     # Curriculum configuration file
│   ├── evaluate.py                # Model evaluation script
│   ├── models/                    # Saved model checkpoints
│   ├── assets/                    # Training assets or resources
│   ├── figs/                      # Figures or training plots
│   └── project_env/               # Environment setup files
│
├── backend/                       # FastAPI backend application
│
├── web/                           # React frontend application
│
├── deployment/                    # AWS SageMaker deployment scripts
│
├── requirements.txt               # Root project dependencies
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SHaruto0/Connect4-RL.git
cd Connect4-RL
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd web
npm install
cd ..
```

---

## Training

Run the training script to start the curriculum learning process:

```bash
python RL/main.py
```

The curriculum is defined in `RL/curriculum_config.json`:

```json
{
  "curriculum": [
    {
      "stage": 0,
      "mode": "random",
      "episodes": 5000,
      "saveas": "models/stage0_model.pth"
    },
    {
      "stage": 1,
      "mode": "selfplay",
      "episodes": 5000,
      "saveas": "models/stage1_model.pth",
      "model_path": "models/stage0_model.pth"
    }
  ]
}
```

You can adjust the number of episodes, mode, and model paths as needed.

---

## Tech Stack

### Training

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Training visualization

### Backend

- **FastAPI**: High-performance API framework
- **PyTorch**: Model inference
- **AWS SageMaker**: Model hosting and deployment
- **Boto3**: AWS SDK for Python

### Frontend

- **React**: UI framework
- **TailwindCSS**: Styling
- **Axios**: HTTP client

### Deployment

- **AWS SageMaker**: Model hosting

---

## Next Steps

The next phase of this project involves building out the complete application stack:

1. **Frontend Development**: Create an interactive React-based Connect 4 game interface with TailwindCSS styling
2. **Backend Development**: Build FastAPI endpoints to handle game logic and communicate with the AI model
3. **Deployment**: Deploy the trained PyTorch model to AWS SageMaker for scalable inference

---

## License

MIT License - see [LICENSE](LICENSE) file for details
