import torch.nn as nn
from torch_geometric.nn import GATConv

from utils import dyn_load


class GAT(nn.Module):

    def __init__(self,
                 num_node_features, hidden_channels, num_heads, num_classes,
                 activation, pooling, dropout
                 ):
        super().__init__()

        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout

        self.conv1 = GATConv(self.num_node_features, self.hidden_channels, heads=self.num_heads)
        self.conv2 = GATConv(self.num_heads * self.hidden_channels, self.hidden_channels, heads=self.num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_heads * self.hidden_channels, self.hidden_channels),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, self.num_classes),
            nn.Sigmoid()
        )
        self.activation = dyn_load(activation)
        self.pool = dyn_load(pooling)

    def forward(self, **inputs):
        x = inputs["x"]
        edge_index = inputs["edge_index"]
        batch = inputs["batch"]
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.pool(x, batch)
        x = self.classifier(x)
        return x


def from_config(**config):
    return GAT(
        int(config["preprocessing"]["emb_dim"]),
        int(config["xp"]["hparams"]["hidden_channels"]),
        int(config["xp"]["hparams"]["num_heads"]),
        int(config["dataset"]["num_classes"]),
        config["xp"]["hparams"]["activation"],
        config["xp"]["hparams"]["pooling"],
        float(config["xp"]["hparams"]["dropout"])
    )


def to_inputs(data, device):
    return {
        "x": data.x.to(device),
        "edge_index": data.edge_index.to(device),
        "batch": data.batch.to(device)
    }
