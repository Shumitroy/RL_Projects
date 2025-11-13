import torch
import random
from collections import deque
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def put(self,transition):
        self.buffer.append(transition)
    def sample(self,n):
        batch = random.sample(self.buffer,n)
        states,actions,rewards,next_states,done_masks = zip(*batch)
        return (torch.tensor(states,dtype=torch.float,device=device),
                torch.tensor(actions,device=device).unsqueeze(1),
                torch.tensor(rewards,dtype=torch.float,device=device).unsqueeze(1),
                torch.tensor(next_states,dtype=torch.float,device=device),
                torch.tensor(done_masks,dtype=torch.float,device=device).unsqueeze(1))
    def size(self): return len(self.buffer)


def train(q_net, q_target, memory, optimizer, batch_size, gamma):
    if memory.size() < batch_size:
        return
    s,a,r,s_next,mask = memory.sample(batch_size)
    q_values = q_net(s).gather(1,a)
    
    next_actions = q_net(s_next).argmax(1,keepdim=True)
    
    q_target_next = q_target(s_next).gather(1,next_actions)
    target = r + gamma * q_target_next * mask
    loss = F.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()