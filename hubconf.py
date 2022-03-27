from torch import flatten
from torch.hub import load
import torch.nn as nn
import torch.nn.functional as F
import os

def model_net(pretrained = False, **kwargs):
  class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

  model = Net(**kwargs)

  if pretrained:
    dirname = os.path.dirname(__file__)
    checkpoint = os.path.join(dirname, 'weights/save.pth')
    state_dict = load(checkpoint)
    model.load_state_dict(state_dict)

  return model
