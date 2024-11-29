import torch
from torch_geometric.data import HeteroData
from load_imdb import load_imdb
from torch_scatter import scatter_mean

# Load the IMDB dataset
hg, features, labels, num_labels, train_indices, valid_indices, test_indices, \
    train_mask, valid_mask, test_mask, node_type_names, link_type_dic, label_names = load_imdb(feat_type=0, random_state=42)

print("Original edge types:", hg.edge_types)
print("\nOriginal edge counts:")
for edge_type in hg.edge_types:
    print(f"{edge_type}: {hg[edge_type].edge_index.size(1)}")

# First assign movie features
hg['movie'].x = features

# Then assign features to intermediate nodes
for node_type in ['director', 'actor', 'keyword']:
    related_edges = hg[('movie', f'to_{node_type}', node_type)].edge_index
    # Get connected movie features and average them
    node_feats = scatter_mean(hg['movie'].x[related_edges[0]], related_edges[1], 
                            dim=0, dim_size=hg[node_type].num_nodes)
    hg[node_type].x = node_feats

# Create new graph with only metapath edges
new_hg = HeteroData()

# Copy node features and other attributes
new_hg['movie'].x = hg['movie'].x
new_hg['movie'].y = hg['movie'].y
new_hg['movie'].train_mask = hg['movie'].train_mask
new_hg['movie'].val_mask = hg['movie'].val_mask
new_hg['movie'].test_mask = hg['movie'].test_mask

# Copy intermediate node features
for node_type in ['director', 'actor']:
    new_hg[node_type].x = hg[node_type].x

# Function to find and add metapath edges
def add_metapath_edges(hg_new, hg_orig, start_edge, end_edge, path_name):
    # Get edges for both steps
    start_edges = hg_orig[start_edge].edge_index
    end_edges = hg_orig[end_edge].edge_index
    
    # Create dictionary of intermediate nodes and their connected movies
    inter_to_movies = {}
    for i in range(start_edges.size(1)):
        movie, inter = start_edges[:, i]
        if inter.item() not in inter_to_movies:
            inter_to_movies[inter.item()] = []
        inter_to_movies[inter.item()].append(movie.item())
    
    # Only keep intermediate nodes with at least 2 movies
    valid_inters = {k: v for k, v in inter_to_movies.items() if len(v) >= 2}
    
    # Create new edge lists for both steps
    step1_src, step1_dst = [], []
    step2_src, step2_dst = [], []
    
    # Add edges for valid paths
    for inter, movies in valid_inters.items():
        for movie in movies:
            # Add first step edges
            step1_src.append(movie)
            step1_dst.append(inter)
            
            # Add second step edges
            step2_src.append(inter)
            for other_movie in movies:
                if other_movie != movie:
                    step2_dst.append(other_movie)
    
    # Add edges to new graph if we found any
    if step1_src:
        edge_index_step1 = torch.tensor([step1_src, step1_dst])
        edge_index_step2 = torch.tensor([step2_src, step2_dst])
        
        src_type, _, dst_type = start_edge
        mid_type = dst_type
        
        # Add edges with new type names indicating metapath step
        hg_new[(src_type, f'{path_name}_step1', mid_type)].edge_index = edge_index_step1
        hg_new[(mid_type, f'{path_name}_step2', src_type)].edge_index = edge_index_step2
        
        return edge_index_step1.size(1), edge_index_step2.size(1)
    return 0, 0

# Add MAM metapath edges
mam_step1, mam_step2 = add_metapath_edges(
    new_hg, 
    hg,
    ('movie', 'to_actor', 'actor'),
    ('actor', 'to_movie', 'movie'),
    'MAM'
)

# Add MDM metapath edges
mdm_step1, mdm_step2 = add_metapath_edges(
    new_hg, 
    hg,
    ('movie', 'to_director', 'director'),
    ('director', 'to_movie', 'movie'),
    'MDM'
)

# Print statistics to verify
print("\nMetapath Statistics:")
print(f"MAM paths - Step 1 edges: {mam_step1}, Step 2 edges: {mam_step2}")
print(f"MDM paths - Step 1 edges: {mdm_step1}, Step 2 edges: {mdm_step2}")

print("\nNew edge types:", new_hg.edge_types)
print("\nNew edge counts:")
for edge_type in new_hg.edge_types:
    print(f"{edge_type}: {new_hg[edge_type].edge_index.size(1)}")

print("\nNode feature sizes:")
for node_type in new_hg.node_types:
    print(f"{node_type}: {new_hg[node_type].x.shape}")

# Verify some paths
def print_sample_paths(hg, path_type, num_samples=5):
    step1_edges = hg[('movie', f'{path_type}_step1', path_type.split('M')[1].lower())].edge_index
    step2_edges = hg[(path_type.split('M')[1].lower(), f'{path_type}_step2', 'movie')].edge_index
    
    print(f"\nSample {path_type} paths:")
    seen_paths = set()
    count = 0
    for i in range(step1_edges.size(1)):
        start_movie = step1_edges[0, i].item()
        inter_node = step1_edges[1, i].item()
        
        # Find where this intermediate node appears in step 2
        step2_indices = (step2_edges[0] == inter_node).nonzero().view(-1)
        for idx in step2_indices:
            end_movie = step2_edges[1, idx].item()
            if start_movie != end_movie:
                path_key = (start_movie, inter_node, end_movie)
                if path_key not in seen_paths:
                    print(f"Movie {start_movie} -> {path_type.split('M')[1]} {inter_node} -> Movie {end_movie}")
                    seen_paths.add(path_key)
                    count += 1
                    if count >= num_samples:
                        break
        if count >= num_samples:
            break

print_sample_paths(new_hg, 'MAM')
print_sample_paths(new_hg, 'MDM')