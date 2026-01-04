import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, n_action):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, fc1_dim, bias=True)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim, bias=True)
        self.output = nn.Linear(fc2_dim, n_action, bias=True)

        # Xavier weights initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.output.weight)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x
    
if __name__ == "__main__":
    model = DQN(1,2,3,5)
    print(model)