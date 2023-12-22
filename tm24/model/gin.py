import torch
import torch.nn as nn
from torch_geometric.nn import GINConv

from utils import dyn_load


class GIN(nn.Module):

    def __init__(self,
                 num_node_features, hidden_channels, num_classes,
                 activation, pooling
                 ):
        super().__init__()

        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(self.num_node_features, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(),
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU()
        ))
        self.fc = nn.Linear(self.hidden_channels, self.num_classes)
        self.activation = dyn_load(activation)
        self.pool = dyn_load(pooling)

    def forward(self, **inputs):
        x = inputs["x"]
        edge_index = inputs["edge_index"]
        batch = inputs["batch"]

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.pool(x, batch)
        x = self.fc(x)
        return torch.sigmoid(x)


def from_config(**config):
    return GIN(
        int(config["preprocessing"]["emb_dim"]),
        int(config["xp"]["hparams"]["hidden_channels"]),
        int(config["dataset"]["num_classes"]),
        config["xp"]["hparams"]["activation"],
        config["xp"]["hparams"]["pooling"],
    )


def to_inputs(data, device):
    return {
        "x": data.x.to(device),
        "edge_index": data.edge_index.to(device),
        "batch": data.batch.to(device)
    }
