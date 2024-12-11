from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn import GATConv

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
    """The Heterogenous Graph Attention Operator from the
    `"Heterogenous Graph Attention Network"
    <https://arxiv.org/abs/1903.07293>`_ paper.

    .. note::

        For an example of using HANConv, see `examples/hetero/han_imdb.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/han_imdb.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        negative_slope=0.2,
        dropout: float = 0.0,
        metapath_data: Dict = None,  
        node_features: Dict = None,
        metapath_encoder: str = 'multihop',
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.dropout = dropout
        self.k_lin = nn.Linear(out_channels, out_channels)
        self.q = nn.Parameter(torch.empty(1, out_channels))
        self.metapath_data = metapath_data
        self.node_features = node_features
        self.metapath_encoder = metapath_encoder
        self.proj = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj[node_type] = Linear(in_channels, out_channels)

        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for metapath_name in metapath_data.keys():
            edge_type = f'metapath_{metapath_name}'
            self.lin_src[edge_type] = nn.Parameter(torch.empty(1, heads, dim))
            self.lin_dst[edge_type] = nn.Parameter(torch.empty(1, heads, dim))
        self.W_h = nn.Parameter(torch.empty(heads, dim, dim))  # Source transform
        self.W_t = nn.Parameter(torch.empty(heads, dim, dim))  # Target transform 
        self.v_a = nn.Parameter(torch.empty(heads, 2*dim))     # Attention vector


        # Add these parameters to __init__:
        self.W_q = nn.Parameter(torch.empty(heads, out_channels // heads))  # Query projection for source node
        
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin.reset_parameters()
        glorot(self.q)
        glorot(self.W_h)
        glorot(self.W_t)
        glorot(self.v_a)
        glorot(self.W_q)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        return_semantic_attention_weights: bool = False,
    ) -> Union[Dict[NodeType, OptTensor], Tuple[Dict[NodeType, OptTensor],
                                                Dict[NodeType, OptTensor]]]:
        device = next(self.parameters()).device

        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        # Project node features once at start
        for node_type in self.node_features.keys():
            x_node_dict[node_type] = self.proj[node_type](self.node_features[node_type].to(device)).view(-1, H, D)
            out_dict[node_type] = []

        # Process each metapath
        for metapath_name, data in self.metapath_data.items():
            paths = data['paths'].to(device)
            node_types = data['node_types_list']
            dst_type = node_types[0] 
            
            
            if self.metapath_encoder == 'mean':
                x_src = self.mean_encoder(paths, node_types, x_node_dict)
            elif self.metapath_encoder == 'multihop':
                x_src = self.multihop_encoder(paths, node_types, x_node_dict)
            elif self.metapath_encoder == 'gat':
                x_src = self.gat_encoder(paths, node_types, x_node_dict)
            elif self.metapath_encoder == 'direct':
                x_src = self.direct_attention_encoder(paths, node_types, x_node_dict)
            elif self.metapath_encoder == 'multihop2':
                x_src = self.multihop_encoder2(paths, node_types, x_node_dict)
            else:
                raise ValueError('Invalid metapath encoder')
            
            x_dst = x_node_dict[dst_type] 


            # Create edge index for instance aggregation
            edge_index = torch.stack([
                torch.arange(len(paths), device=device),
                paths[:, 0] 
            ])


            # Instance-level attention aggregation
            edge_type = f'metapath_{metapath_name}'
            lin_src = self.lin_src[edge_type].to(device)
            lin_dst = self.lin_dst[edge_type].to(device)
            alpha_src = (x_src * lin_src).sum(dim=-1)
            alpha_dst = (x_dst * lin_dst).sum(dim=-1)
            
            # Propagate to aggregate instances
            out = self.propagate(edge_index, x=(x_src, x_dst),
                    alpha=(alpha_src, alpha_dst))
            
            out = F.relu(out)
            out_dict[dst_type].append(out)

        # Semantic attention across metapaths
        semantic_attn_dict = {}
        for node_type, outs in out_dict.items():
            if len(outs) > 0:  # Only process if we have any metapaths for this node type
                out, attn = group(outs, self.q.to(device), self.k_lin.to(device))
                out_dict[node_type] = out
                semantic_attn_dict[node_type] = attn

        if return_semantic_attention_weights:
            return out_dict, semantic_attn_dict

        return out_dict
    
    def direct_attention_encoder(self,
        paths: Tensor, 
        node_types: List[str],
        x_node_dict: Dict[str, Tensor]
    ) -> Tensor:
        """
        Efficient encoder using source node attention
        """
        # Gather features for all nodes in paths (same as mean_encoder)
        path_features = []
        for pos, node_type in enumerate(node_types):
            node_indices = paths[:, pos]
            pos_features = x_node_dict[node_type].to(paths.device).index_select(0, node_indices.to(paths.device))
            path_features.append(pos_features)
            
        # Stack to get [num_instances, path_length, heads, dim]
        path_features = torch.stack(path_features, dim=1)
        
        # Calculate attention scores from source to all positions
        source_features = path_features[:, 0].unsqueeze(1)  # [num_instances, 1, heads, dim]
        scores = torch.einsum('nihd,hd->nih', source_features, self.W_q)  # [num_instances, 1, heads]
        scores = scores.expand(-1, path_features.size(1), -1)  # [num_instances, path_length, heads]
        scores = scores / math.sqrt(self.out_channels // self.heads)
        
        # Softmax over path length dimension
        attn = F.softmax(scores, dim=1).unsqueeze(-1)  # [num_instances, path_length, heads, 1]
        
        # Weighted sum
        out = (attn * path_features).sum(dim=1)  # [num_instances, heads, dim]
        
        return out
        
    def multihop_encoder2(self,
        paths: Tensor, 
        node_types: List[str],
        x_node_dict: Dict[str, Tensor]
        ) -> Tensor:
        """
        Sequential inward attention for arbitrary length metapaths.
        """
        # Gather features exactly like mean_encoder 
        path_features = []
        for pos, node_type in enumerate(node_types):
            node_indices = paths[:, pos]
            pos_features = x_node_dict[node_type].to(paths.device).index_select(0, node_indices.to(paths.device))
            path_features.append(pos_features)
                
        # Stack to get [num_instances, path_length, heads, dim]
        path_features = torch.stack(path_features, dim=1)
        path_length = path_features.size(1)
        
        # Prepare adjacent pairs
        outer_nodes = path_features[:, 1:]  
        inner_nodes = path_features[:, :-1]
        
        # Simple attention computation  
        attn_logits = torch.sum(outer_nodes * inner_nodes, dim=-1)  
        attn_logits = attn_logits / math.sqrt(self.out_channels // self.heads)
        print(f"\nAttention logits stats - Mean: {attn_logits.mean():.4f}, Std: {attn_logits.std():.4f}")
        
        # Distance decay
        gamma = 0.5
        distance_decay = torch.exp(-gamma * torch.arange(1, path_length, device=paths.device))
        
        # Attention weights with decay
        edge_attn = F.softmax(attn_logits * distance_decay.view(1, -1, 1), dim=1)
        print(f"\nEdge attention stats - Mean: {edge_attn.mean():.4f}, Std: {edge_attn.std():.4f}")
        
        # Cumulative attention flow
        cum_attn = torch.cumprod(edge_attn, dim=1)
        print(f"\nCumulative attention stats - Mean: {cum_attn.mean():.4f}, Std: {cum_attn.std():.4f}")
        
        # Final aggregation
        out = path_features[:, 0] + (cum_attn.unsqueeze(-1) * path_features[:, 1:]).sum(dim=1)
        
        return out
    def multihop_encoder(self,
        paths: Tensor,
        node_types: List[str],
        x_node_dict: Dict[str, Tensor]
    ) -> Tensor:
        device = paths.device
        num_instances, path_length = paths.size()
        H, D = self.heads, self.out_channels // self.heads
        gamma = torch.tensor(0.1)  # Fixed gamma value for decay
        chunk_size = 1024  # Process in chunks to save memory

        outputs = []
        for start_idx in range(0, num_instances, chunk_size):
            end_idx = min(start_idx + chunk_size, num_instances)
            paths_chunk = paths[start_idx:end_idx]
            
            # Get path features
            path_features = []
            for pos, node_type in enumerate(node_types):
                node_indices = paths_chunk[:, pos]
                features = x_node_dict[node_type].to(device).index_select(0, node_indices)
                path_features.append(features)
            path_features = torch.stack(path_features, dim=1)
            """
            with torch.no_grad():
                print(f"\nProcess path of length {path_length}:")
                print(f"Node types: {node_types}")
                print(f"Feature shape: {path_features.shape}")
            """
            # Compute pairwise attentions - flipped for inward flow
            h_dst = path_features[:, :-1]  # nodes receiving messages
            h_src = path_features[:, 1:]   # nodes sending messages
            
            # Transform features
            h_src_t = torch.einsum('nphd,hde->nphe', h_src, self.W_h)
            h_dst_t = torch.einsum('nphd,hde->nphe', h_dst, self.W_t)
            
            # Compute attention scores flowing inward
            concat = torch.cat([h_dst_t, h_src_t], dim=-1)  
            attn_logits = torch.einsum('nphd,hd->nph', torch.tanh(concat), self.v_a)
            edge_attn = torch.sigmoid(F.leaky_relu(attn_logits, self.negative_slope))
            if self.training:
                edge_attn = F.dropout(edge_attn, p=0.1, training=self.training)
            """
            with torch.no_grad():
                print("\nPairwise attention scores:")
                print(f"Mean: {edge_attn.mean():.4f}, Std: {edge_attn.std():.4f}")
                for i in range(path_length-1):
                    print(f"Edge {i+1}->{i}: {edge_attn[:,i].mean():.4f} ± {edge_attn[:,i].std():.4f}")
            """
            # Initialize output with first node (source node)
            chunk_output = path_features[:, 0]
            
            # Handle k>0 cases
            for k in range(1, path_length):
                # Compute exponential decay (close nodes still weighted higher)
                decay = torch.exp(-gamma * k)
                
                # Get attention scores along path back to source
                if k == 1:
                    k_attentions = edge_attn[:, 0:1]  # just 1→0
                else:
                    k_attentions = torch.flip(edge_attn[:, :k], dims=[1])
                # Compute product of attentions along path
                attn_prod = torch.prod(k_attentions, dim=1)
                
                # Combine decay with attention product
                coef = decay * attn_prod
                """"
                with torch.no_grad():
                    print(f"\nPosition {k}:")
                    print(f"Decay term (e^(-{gamma}*{k})): {decay:.4f}")
                    print(f"Attention product - Mean: {attn_prod.mean():.4f}, Std: {attn_prod.std():.4f}")
                    print(f"Final coef - Mean: {coef.mean():.4f}, Std: {coef.std():.4f}")
                """
                # Add contribution
                chunk_output = chunk_output + coef.unsqueeze(-1) * path_features[:, k]
            
            outputs.append(chunk_output)
            del path_features, h_dst, h_src, h_src_t, h_dst_t, concat, attn_logits, edge_attn
            torch.cuda.empty_cache()

        return torch.cat(outputs, dim=0)
    
    def reset_parameters(self):
        super().reset_parameters()
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin.reset_parameters()
        glorot(self.q)
        # Initialize attention parameters with larger values
        std = 1.0 / math.sqrt(self.out_channels)
        self.W_h.data.normal_(0, std)
        self.W_t.data.normal_(0, std)
        self.v_a.data.normal_(0, std)
    def mean_encoder(self,
        paths: Tensor,
        node_types: List[str],
        x_node_dict: Dict[str, Tensor]
    ) -> Tensor:
        """
        Encodes metapath instances using mean pooling over node features.

        Args:
            paths (Tensor): Tensor of shape [num_instances, path_length], 
                            containing indices of nodes in the metapath.
            node_types (List[str]): List of node types in the metapath.
            x_node_dict (Dict[str, Tensor]): Dictionary mapping node types to 
                                            their projected feature tensors.

        Returns:
            Tensor: Encoded metapath instance features of shape 
                    [num_instances, heads, dim].
        """
        # Gather features for all nodes in paths
        path_features = []
        for pos, node_type in enumerate(node_types):
            node_indices = paths[:, pos]
            pos_features = x_node_dict[node_type].to(paths.device).index_select(0, node_indices.to(paths.device))
            path_features.append(pos_features)
        # Stack and mean pool [num_instances, path_length, heads, dim]
        path_features = torch.stack(path_features, dim=1)
        return torch.mean(path_features, dim=1)  # [num_instances, heads, dim]

    def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')