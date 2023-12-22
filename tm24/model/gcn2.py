from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

from utils import dyn_load

class GCN2(nn.Module):
    
    
    def __init__(self,
        num_node_features, hidden_channels, num_classes,
        activation, pooling, dropout
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = GCNConv(self.num_node_features, self.hidden_channels)
        self.norm1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.norm2 = BatchNorm(hidden_channels)
        self.classifier = nn.Linear(self.hidden_channels, self.num_classes)
        self.activation = dyn_load(activation)
        self.pool = dyn_load(pooling)


    def forward(self, **inputs):
        
        x = inputs["x"]
        edge_index = inputs["edge_index"]
        batch = inputs["batch"]
        
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pool(x, batch)
        
        x = self.classifier(x)
        return torch.sigmoid(x)


def from_config(**config):
    return GCN2(
        int(config["preprocessing"]["emb_dim"]),
        int(config["xp"]["hparams"]["hidden_channels"]),
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