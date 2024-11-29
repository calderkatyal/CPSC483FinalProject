import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from typing import List, Tuple, Dict, Optional
from torch import Tensor
from MetaPathAdd import MetaPathAdd


# Define metapaths
metapaths = [
    [('movie', 'to_director', 'director'), ('director', 'to_movie', 'movie')],  # MDM
    [('movie', 'to_actor', 'actor'), ('actor', 'to_movie', 'movie')]            # MAM
]

# Manually create the small heterogeneous graph
def create_manual_graph():
    data = HeteroData()
    
    # Create node features for 'movie', 'director', 'actor'
    # For simplicity, we'll use small feature vectors
    data['movie'].x = torch.tensor([
        [0.1, 0.2],  # movie 0
        [0.3, 0.4],  # movie 1
        [0.5, 0.6],  # movie 2
        [0.7, 0.8],  # movie 3
        [0.9, 1.0],  # movie 4
    ], dtype=torch.float)
    
    data['director'].x = torch.tensor([
        [1.1, 1.2],  # director 0
        [1.3, 1.4],  # director 1
        [1.5, 1.6],  # director 2
    ], dtype=torch.float)
    
    data['actor'].x = torch.tensor([
        [2.1, 2.2],  # actor 0
        [2.3, 2.4],  # actor 1
        [2.5, 2.6],  # actor 2
    ], dtype=torch.float)
    
    # Manually create edge indices
    # For 'movie' to 'director'
    data[('movie', 'to_director', 'director')].edge_index = torch.tensor([
        [0, 1, 2],  # movie indices
        [0, 1, 2],  # director indices
    ], dtype=torch.long)
    data[('director', 'to_movie', 'movie')].edge_index = torch.tensor([
        [0, 1, 2],  # director indices
        [1, 2, 3],  # movie indices
    ], dtype=torch.long)
    
    # For 'movie' to 'actor'
    data[('movie', 'to_actor', 'actor')].edge_index = torch.tensor([
        [0, 1, 3],  # movie indices
        [0, 1, 2],  # actor indices
    ], dtype=torch.long)
    data[('actor', 'to_movie', 'movie')].edge_index = torch.tensor([
        [0, 1, 2],  # actor indices
        [2, 3, 4],  # movie indices
    ], dtype=torch.long)
    
    return data

# Create the manual graph
data = create_manual_graph()

print("\nBefore metapath transform:")
print(f"Node types: {data.node_types}")
print(f"Edge types: {data.edge_types}")
for node_type in data.node_types:
    print(f"Number of {node_type} nodes: {data[node_type].num_nodes}")


# Apply the metapath transform
transform = MetaPathAdd(
    metapaths=metapaths,
    drop_orig_edge_types=False,
    drop_unconnected_node_types=False
)
data, metapath_data = transform(data)

import pprint

def print_metapath_data(metapath_data):
    formatted_data = {}
    for metapath_name, meta_data in metapath_data.items():
     
        metapath_key = metapath_name  # Fallback

        formatted_data[metapath_key] = {
            'paths': meta_data['paths'],
            'features': meta_data['features']
        }

    pprint.pprint(formatted_data)

print("\nFormatted Metapath Data:")
print_metapath_data(metapath_data)
