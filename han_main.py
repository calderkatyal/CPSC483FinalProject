import os.path as osp
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn.conv import HANConv
from load_imdb import load_imdb, data_loader
import numpy as np

# Load the IMDB dataset
hg, features, labels, num_labels, train_indices, valid_indices, test_indices, \
    train_mask, valid_mask, test_mask, node_type_names, link_type_dic, label_names = load_imdb(feat_type=0, random_state=42)

assert (train_mask & valid_mask).sum() == 0, "Train and validation masks overlap!"
assert (train_mask & test_mask).sum() == 0, "Train and test masks overlap!"
assert (valid_mask & test_mask).sum() == 0, "Validation and test masks overlap!"

print(f"Training samples: {train_mask.sum() / len(train_mask) * 100}%")
print(f"Validation samples: {valid_mask.sum() / len(valid_mask) * 100}%")
print(f"Test samples: {test_mask.sum() / len(test_mask) * 100}%")

# Define metapaths based on your link_type_dic
metapaths = [
    [('movie', 'to_director', 'director'), ('director', 'to_movie', 'movie')],  # MDM
    [('movie', 'to_actor', 'actor'), ('actor', 'to_movie', 'movie')]           # MAM
]

# Print initial graph stats
print("\nBefore metapath transform:")
print(f"Node types: {hg.node_types}")
print(f"Edge types: {hg.edge_types}")
for node_type in hg.node_types:
    print(f"Number of {node_type} nodes: {hg[node_type].num_nodes}")

# Apply metapath transform
transform = T.AddMetaPaths(
    metapaths=metapaths,
    drop_orig_edge_types=True,  
    drop_unconnected_node_types=True 
)
hg = transform(hg)

# Print transformed graph stats
print("\nAfter metapath transform:")
print(f"Node types: {hg.node_types}")
print(f"Edge types: {hg.edge_types}")
for node_type in hg.node_types:
    print(f"Number of {node_type} nodes: {hg[node_type].num_nodes}")

# Assign features only to movie nodes
hg['movie'].x = features
hg['movie'].y = labels
hg['movie'].train_mask = train_mask
hg['movie'].val_mask = valid_mask
hg['movie'].test_mask = test_mask

# Calculate features for other nodes
for node_type in ['director', 'actor', 'keyword']:
    related_edges = hg[('movie', f'to_{node_type}', node_type)].edge_index
    node_feats = scatter_mean(features[related_edges[0]], related_edges[1], 
                            dim=0, dim_size=hg[node_type].num_nodes)
    hg[node_type].x = node_feats

# Store additional metadata
hg.num_labels = num_labels
hg.train_indices = train_indices
hg.valid_indices = valid_indices
hg.test_indices = test_indices
hg.node_type_names = node_type_names
hg.link_type_dic = link_type_dic
hg.label_names = label_names

# Define HAN model
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

# Initialize model with only movie features
model = HAN(in_channels={'movie': features.shape[1]},
            out_channels=num_labels)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    loss = F.binary_cross_entropy_with_logits(out[mask], hg['movie'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)

# Testing function
@torch.no_grad()
def test() -> Dict[str, List[float]]:
    model.eval()
    logits = model(hg.x_dict, hg.edge_index_dict)
    pred_probs = torch.sigmoid(logits)
    pred = (pred_probs > 0.5).float()

    # Handle cases where no predictions are made
    no_pred_mask = pred.sum(dim=1) == 0
    max_indices = torch.argmax(pred_probs, dim=1)
    pred[no_pred_mask, :] = 0
    pred[no_pred_mask, max_indices[no_pred_mask]] = 1

    metrics = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = hg['movie'][split]
        y_true = labels[mask]
        y_pred = pred[mask]
        
        # Calculate Micro F1
        true_pos = (y_true * y_pred).sum(dim=0)
        pred_pos = y_pred.sum(dim=0)
        actual_pos = y_true.sum(dim=0)
        
        micro_precision = true_pos.sum() / (pred_pos.sum() + 1e-8)
        micro_recall = true_pos.sum() / (actual_pos.sum() + 1e-8)
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-8)
        
        # Calculate Macro F1
        precision = true_pos / (pred_pos + 1e-8)
        recall = true_pos / (actual_pos + 1e-8)
        f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
        macro_f1 = f1_per_class[~torch.isnan(f1_per_class)].mean()

        # Print example incorrect prediction for test set
        if split == 'test_mask':
            incorrect = ~(y_true == y_pred).all(dim=1)
            if incorrect.any():
                idx = torch.where(incorrect)[0][torch.randint(0, incorrect.sum(), (1,))].item()
                true_labs = [label_names[i] for i in range(len(label_names)) if y_true[idx][i].item() == 1]
                pred_labs = [label_names[i] for i in range(len(label_names)) if y_pred[idx][i].item() == 1]
                probs = pred_probs[mask][idx]
                
                print("\nRandom Incorrect Prediction:")
                print("True:", true_labs)
                print("Pred:", pred_labs)
                print("\nAll probabilities:")
                for lab, prob in zip(label_names, probs):
                    print(f"{lab}: {prob:.4f}")
        
        metrics.append({
            'micro_f1': float(micro_f1),
            'macro_f1': float(macro_f1)
        })

    print(f'Micro F1 -> Train: {metrics[0]["micro_f1"]:.4f}, Val: {metrics[1]["micro_f1"]:.4f}, Test: {metrics[2]["micro_f1"]:.4f}')
    print(f'Macro F1 -> Train: {metrics[0]["macro_f1"]:.4f}, Val: {metrics[1]["macro_f1"]:.4f}, Test: {metrics[2]["macro_f1"]:.4f}')
    
    return {'micro_f1': [m['micro_f1'] for m in metrics],
            'macro_f1': [m['macro_f1'] for m in metrics]}

# Training loop with early stopping
best_val_micro_f1 = 0
best_val_macro_f1 = 0
best_state = None
start_patience = patience = 100

for epoch in range(1, 200):
    print(f'\nEpoch {epoch}:')
    loss = train()
    print(f'Train Loss: {loss:.4f}')
    metrics = test()
    
    val_micro_f1 = metrics['micro_f1'][1]
    val_macro_f1 = metrics['macro_f1'][1]
    
    if best_val_macro_f1 <= val_macro_f1:
        patience = start_patience
        best_val_macro_f1 = val_macro_f1
        best_val_micro_f1 = val_micro_f1
        best_state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_macro_f1': val_macro_f1,
            'val_micro_f1': val_micro_f1,
            'test_macro_f1': metrics['macro_f1'][2],
            'test_micro_f1': metrics['micro_f1'][2]
        }
    else:
        patience -= 1

    if patience <= 0:
        print(f'\nEarly stopping after {epoch} epochs!')
        print(f'Best validation macro F1: {best_val_macro_f1:.4f}')
        print(f'Best validation micro F1: {best_val_micro_f1:.4f}')
        print(f'Corresponding test macro F1: {best_state["test_macro_f1"]:.4f}')
        print(f'Corresponding test micro F1: {best_state["test_micro_f1"]:.4f}')
        break

# Load best model
if best_state is not None:
    model.load_state_dict(best_state['model_state'])