import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    'DQN actor (policy) model with option to choose dueling architecture'
    
    def __init__(self, state_size, action_size, seed, num_layers, hidden_units, duel=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            num_layers (int): Number of hidden layers
            hidden_units (int): Number of nodes in each hidden layer
            duel (bool): Whether to use dueling network (default=True)
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.duel = duel
        self.input = nn.Linear(state_size, hidden_units)
        self.layers = nn.ModuleList([nn.Linear(hidden_units, hidden_units) for _ in range(num_layers)])
        if duel:
            # the final FC layer in DQN becomes split into 2 streams of 2 FC layers in dueling
            self.val_fc_input = nn.Linear(hidden_units, int(hidden_units/2))
            self.val_fc_output = nn.Linear(int(hidden_units/2), 1)
            self.adv_fc_input = nn.Linear(hidden_units, int(hidden_units/2))
            self.adv_fc_output = nn.Linear(int(hidden_units/2), action_size)
        else:
            self.output = nn.Linear(hidden_units, action_size)
        
    def forward(self, state):
        """Forward pass that maps state -> action values."""
        x = F.relu(self.input(state))
        for layer in self.layers:
            x = F.relu(layer(x))
        if self.duel:
            # Value function estimator
            val = F.relu(self.val_fc_input(x))
            val = self.val_fc_output(val)
            # Advantage function estimator
            adv = F.relu(self.adv_fc_input(x))
            adv = self.adv_fc_output(adv)
            # Subtract mean so that V and A are uniquely identifiable for a given Q
            return val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.action_size)
        else:
            return self.output(x)
    