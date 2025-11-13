import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Qnet(nn.Module):
    def __init__(self, action_size, state_size):
        super(Qnet,self).__init__()
        self.fc1 = nn.Linear(state_size,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,action_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def sample_action(self, state, epsilon):
        
        if random.random() < epsilon:
            return random.randint(0,self.fc3.out_features-1)
        else:
            with torch.no_grad():
                return self.forward(state).argmax().item()