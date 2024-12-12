import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from config import encoder_config

class CustomGATEncoder(nn.Module):
    """
    Graph attention based encoder module. This encoder consumes a TensorDict object
    which contains the graph node coordinates. The purpose of the encoder is to generate embeddings
    for these coordiantes that will eventually be consumed by the decoder. The encoder architecture is:
    x --> (initial embedding layers) --> [GAT --> BN --> RELU --> dropout]....[GAT --> BN --> RELU --> dropout] --> GAT --> h
    The initial embedding layer is defined as a separate class and is meant to transform the 2D coordinate locations into embedding space. 
    """ 
    def __init__(self, input_dim, hidden_dim, num_layers=3, num_heads = 1, p_dropout = 0.5, loc_embed_intermediate_dim = 64, loc_embed_p_dropout = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.p_dropout = p_dropout
        self.loc_embed = LocationEmbedding(input_dim, hidden_dim, intermediate_dim=loc_embed_intermediate_dim, num_layers=1, p_dropout=loc_embed_p_dropout)
        self.layers = nn.ModuleList([GATConv(hidden_dim, hidden_dim, num_heads, concat=True, edge_dim=2) for _ in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim*num_heads) for _ in range(num_layers - 1)])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, td):
        locs = td["locs"]
        # Transform the 2D coordinates to the embedding dimension
        loc_embedding = self.loc_embed(locs)
        batch_size, num_nodes, _ = loc_embedding.shape

        # Convert from [batch_size, num_nodes, hidden_dim] to [batch_size*num_nodes, hidden_dim]
        x = loc_embedding.reshape(-1, self.hidden_dim)

        # Note that in TSP we are dealing with fully connected graphs. Thus we can use the same edge_index for 
        # every single graph in our batch by constructing one edge_index and repeating them. 
        edge_index = self.build_fully_connected_edge_index(batch_size, num_nodes, self_loops = True)
        for layer, bn in zip(self.layers[:-1], self.bns[:-1]):
            x = layer(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.p_dropout)
        
        x = self.layers[-1](x, edge_index)
        x = x.view(batch_size, num_nodes, self.hidden_dim)

        return x, None

    def build_fully_connected_edge_index(self, batch_size, num_nodes, self_loops = False):
        """
        This function constructs the edge index for a fully connected graph with NUM_NODES nodes. 
        To support batching (and because all graphs in the batch are fully conencted with the same number of nodes),
        this function then repeats the edge_index across the whole batch. 
        """ 
        row = torch.arange(num_nodes).repeat_interleave(num_nodes)
        col = torch.arange(num_nodes).repeat(num_nodes)
        if not self_loops:
            mask = row != col 
            row, col = row[mask], col[mask]
        edge_index = torch.stack([row, col], dim=0).to(self.device)

        batch_edge_index = edge_index.repeat(1, batch_size)

        return batch_edge_index


class LocationEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, intermediate_dim=64, num_layers=3, p_dropout=0.1):
        """
        This module serves as a relatively straightforwrad way to embed the initial location coordinates 
        of INPUT_DIM into the embedding dimension of HIDDEN_DIM. It operatesas:
        x --> (Linear) --> (ReLU) --> (Linear) --> (BN) --> (ReLU) --> (Dropout) --> Linear
        """
        super(LocationEmbedding, self).__init__()
        self.input_layer = nn.Linear(input_dim, intermediate_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(intermediate_dim, intermediate_dim) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(intermediate_dim, hidden_dim)
        self.norms = nn.ModuleList([nn.BatchNorm1d(intermediate_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p_dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size, num_nodes, input_dim = x.shape
        x = x.view(-1, input_dim)

        x = self.input_layer(x)
        x = self.activation(x)
        for layer, norm in zip(self.hidden_layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.output_layer(x)

        # Reshape back to [batch_size, num_nodes, hidden_dim]
        x = x.view(batch_size, num_nodes, -1)
        return x
