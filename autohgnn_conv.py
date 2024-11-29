# autohgnn_conv.py

import torch
import torch_geometric as pyg
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax
from typing import Dict, List, Optional, Tuple, Union
from torch_scatter import scatter_mean

def group(
    xs: List[Tensor],
    q: nn.Parameter,
    k_lin: nn.Module,
) -> Tuple[OptTensor, OptTensor]:

    if len(xs) == 0:
        return None, None
    else:
        num_edge_types = len(xs)
        out = torch.stack(xs)
        if out.numel() == 0:
            return out.view(0, out.size(-1)), None
        attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
        attn = F.softmax(attn_score, dim=0)
        out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
        return out, attn


class AutoHGNNConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        metadata: Optional[tuple] = None,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Feature transformations for each node type
        self.linear_dict = nn.ModuleDict({
            node_type: nn.Linear(dim, out_channels * heads)
            for node_type, dim in in_channels.items()
        })
        
        # Attention vectors (one per head)
        self.attention = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # Semantic-level attention
        self.semantic_attention = nn.Parameter(torch.Tensor(1, out_channels * heads))
        
        self.reset_parameters()

        
    def reset_parameters(self):
        for lin in self.linear_dict.values():
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.xavier_uniform_(self.attention)
        nn.init.xavier_uniform_(self.semantic_attention)
        
    def forward(self, x_dict, edge_index_dict, metapath_dict, graph):
        # Transform node features
        transformed_features = {
            node_type: self.linear_dict[node_type](feat)
            for node_type, feat in x_dict.items()
        }
        
        metapath_outputs = []
        
        # Process each metapath
        for metapath_key, metapath_indices in metapath_dict.items():
            if metapath_indices.size(0) == 0:
                continue
                
            # Get node features along the metapath
            path_features = []
            for i in range(metapath_indices.size(1)):
                node_idx = metapath_indices[:, i]
                # Get node type from graph metadata
                node_type = graph[('movie', metapath_key, 'movie')].node_types[i]
                node_features = transformed_features[node_type][node_idx]
                path_features.append(node_features)
            
            # Mean pooling of node features (simplified encoder)
            metapath_embedding = torch.stack(path_features, dim=1).mean(dim=1)
            
            # Reshape for multi-head attention
            metapath_embedding = metapath_embedding.view(-1, self.heads, self.out_channels)
            
            # Node-level attention
            attn_scores = (metapath_embedding * self.attention).sum(dim=-1)
            attn_weights = softmax(attn_scores, metapath_indices[:, 0])
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention
            out = metapath_embedding * attn_weights.view(-1, self.heads, 1)
            
            # Aggregate per target node
            target_nodes = metapath_indices[:, -1]
            out = scatter_mean(out, target_nodes, dim=0, dim_size=x_dict['movie'].size(0))
            metapath_outputs.append(out)
        
        if not metapath_outputs:
            return {
                'movie': torch.zeros((x_dict['movie'].size(0), self.heads * self.out_channels),
                                  device=x_dict['movie'].device)
            }
            
        # Stack outputs from different metapaths
        metapath_outputs = torch.stack(metapath_outputs, dim=1)  # [num_nodes, num_metapaths, heads, out_channels]
        
        # Semantic-level attention
        semantic_attn = F.softmax(
            (metapath_outputs.mean(dim=2) * self.semantic_attention.view(1, 1, -1)).sum(dim=-1),
            dim=1
        )
        
        # Final aggregation across metapaths
        out = (metapath_outputs * semantic_attn.view(-1, semantic_attn.size(1), 1, 1)).sum(dim=1)
        out = out.view(-1, self.heads * self.out_channels)
        
        return {'movie': out}


    def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)