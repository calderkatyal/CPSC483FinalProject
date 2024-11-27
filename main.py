import os.path as osp
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import HANConv
from load_imdb import load_imdb, data_loader
import numpy as np
from torch_scatter import scatter_mean

# Load the IMDB dataset
hg, features, labels, num_labels, train_indices, valid_indices, test_indices, \
    train_mask, valid_mask, test_mask, node_type_names, link_type_dic, label_names = load_imdb(feat_type=0, random_state=42)


print(f"Training samples: {train_mask.sum() / len(train_mask) * 100}%")
print(f"Validation samples: {valid_mask.sum() / len(valid_mask) * 100}%")
print(f"Test samples: {test_mask.sum() / len(test_mask) * 100}%")

# Define metapaths based on your link_type_dic
metapaths = [
    [('movie', 'to_director', 'director'), ('director', 'to_movie', 'movie')],  # MDM
    [('movie', 'to_actor', 'actor'), ('actor', 'to_movie', 'movie')],          # MAM
    [('movie', 'to_keyword', 'keyword'), ('keyword', 'to_movie', 'movie')]     # MKM
]

# Apply metapath transform
transform = T.AddMetaPaths(
    metapaths=metapaths,
    drop_orig_edge_types=False,  # Keep original edges along with metapath edges
    drop_unconnected_node_types=False  # Keep all node types
)
hg = transform(hg)

# Assign features to each node type (movie, director, actor, keyword)
hg['movie'].x = features  # Original movie features

""""
for node_type in ['director', 'actor', 'keyword']:
    hg[node_type].x = torch.zeros((hg[node_type].num_nodes, features.shape[1]))  # Placeholder features
"""
# CHANGE: Instead of using placeholder features, we will use the average of connected movie features
for node_type in ['director', 'actor', 'keyword']:
    related_edges = hg[('movie', f'to_{node_type}', node_type)].edge_index
    # Get connected movie features and average them
    node_feats = scatter_mean(hg['movie'].x[related_edges[0]], related_edges[1], 
                            dim=0, dim_size=hg[node_type].num_nodes)
    hg[node_type].x = node_feats

# Assign labels and masks
hg['movie'].y = labels.argmax(dim=1).long()  # Convert one-hot encoded labels to class indices
hg['movie'].train_mask = train_mask          # Training mask for 'movie' nodes
hg['movie'].val_mask = valid_mask            # Validation mask for 'movie' nodes
hg['movie'].test_mask = test_mask            # Test mask for 'movie' nodes

# Store additional metadata
hg.num_labels = num_labels                   # Number of unique labels
hg.train_indices = train_indices             # Training indices
hg.valid_indices = valid_indices             # Validation indices
hg.test_indices = test_indices               # Test indices
hg.node_type_names = node_type_names         # Node type names dictionary
hg.link_type_dic = link_type_dic             # Link type dictionary
hg.label_names = label_names                 # List of label names

# Print each attribute with sensible formatting
#print("Heterogeneous Graph (hg):", hg)
print("Node Features (features) shape:", features.shape)
print("Labels (labels) shape:", labels.shape)
print("Number of Labels (num_labels):", num_labels)
print("Training Indices (train_indices) shape:", train_indices.shape)
print("Validation Indices (valid_indices) shape:", valid_indices.shape)
print("Testing Indices (test_indices) shape:", test_indices.shape)
print("Training Mask (train_mask) shape:", train_mask.shape)
print("Validation Mask (valid_mask) shape:", valid_mask.shape)
print("Testing Mask (test_mask) shape:", test_mask.shape)

print("Node Type Names (node_type_names):", node_type_names)
print("Link Type Dictionary (link_type_dic):", link_type_dic)
print("Label Names (label_names):", label_names)


# Print node types with features and their shapes
print("\nNode Types with Features and their Shapes:")
for node_type_id, node_type_name in node_type_names.items():
    if node_type_name in hg.node_types and 'x' in hg[node_type_name]:
        print(f"  - {node_type_name}: features shape {hg[node_type_name]['x'].shape}")
    else:
        print(f"  - {node_type_name}: No features available")

# Define HAN model using your IMDB dataset
class HAN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=128, heads=8):
        super().__init__()
        self.han_conv = HANConv(
            in_channels, hidden_channels, heads=heads,
            dropout=0.6, metadata=hg.metadata()
        )

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['movie'])
        return out

# Initialize the model
model = HAN(in_channels={'movie': features.shape[1], 'director': features.shape[1],
                         'actor': features.shape[1], 'keyword': features.shape[1]},
            out_channels=num_labels)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
hg, model = hg.to(device), model.to(device)

# Initialize lazy modules
with torch.no_grad():
    out = model(hg.x_dict, hg.edge_index_dict)

# Optimizer setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

# Training function
def train() -> float:
    model.train()
    optimizer.zero_grad()
    out = model(hg.x_dict, hg.edge_index_dict)
    mask = hg['movie'].train_mask
    loss = F.cross_entropy(out[mask], hg['movie'].y[mask])
    loss.backward()
    optimizer.step()
    
    print(f'Train Loss: {loss.item()}')  # Print training loss
    return float(loss)

# Testing function
@torch.no_grad()
def test() -> List[float]:
    model.eval()
    pred = model(hg.x_dict, hg.edge_index_dict).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = hg['movie'][split]
        acc = (pred[mask] == hg['movie'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    print(f'Test Accuracies: {accs}')  # Print test accuracies for each split
    return accs

# Training loop with early stopping
best_val_acc = 0
start_patience = patience = 100
for epoch in range(1, 200):
    print(f'Epoch {epoch}:')  # Print epoch number
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    if best_val_acc <= val_acc:
        patience = start_patience
        best_val_acc = val_acc
    else:
        patience -= 1

    if patience <= 0:
        print('Stopping training as validation accuracy did not improve '
              f'for {start_patience} epochs')
        break

