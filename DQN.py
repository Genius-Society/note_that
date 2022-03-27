import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN:
    def __init__(self, layers, lr=0.0005, optim_method=optim.Adam):
        self.layers = layers
        self.lr = lr
        self.loss = F.mse_loss
        self.optim_method = optim_method
        self.TargetNetwork = None
        self.EstimateNetwork = None
        self.optimizer = None
        self.build_model()

    def build_model(self):
        def init_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

        self.EstimateNetwork = nn.Sequential(*self.layers)
        self.EstimateNetwork.apply(init_weights)

        layers_for_target = copy.deepcopy(self.layers)
        self.TargetNetwork = nn.Sequential(*layers_for_target)
        self.TargetNetwork.load_state_dict(self.EstimateNetwork.state_dict())

        self.optimizer = self.optim_method(
            self.EstimateNetwork.parameters(), lr=self.lr)

    def Q_target(self, inp):
        return self.TargetNetwork(inp)

    def Q_estimate(self, inp):
        return self.EstimateNetwork(inp)

    def update_target(self):
        self.TargetNetwork.load_state_dict(self.EstimateNetwork.state_dict())

    def update_parameters(self, estimated, targets):
        loss = self.loss(estimated, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.EstimateNetwork.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, name):
        torch.save(self.EstimateNetwork, name)
        print('------ Model saved ------')
