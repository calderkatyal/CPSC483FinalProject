from load_imdb import load_imdb
from load_imdb import data_loader
import torch as th
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.colors as mcolors



def printInfo(output_file):
    hg, features, labels, num_labels, train_indices, valid_indices, test_indices, \
    train_mask, valid_mask, test_mask, node_type_names, link_type_dic, label_names = load_imdb(feat_type=0, random_state=42)
    
    dl = data_loader('./data/IMDB')
    
    with open(output_file, 'w') as f:
        # Print basic graph information
        total_nodes = sum(hg[node_type].num_nodes for node_type in hg.node_types)
        total_edges = sum(hg[edge_type].edge_index.size(1) for edge_type in hg.edge_types)
        f.write(f"Graph has {total_nodes} total nodes.\n")
        f.write(f"Graph has {total_edges} total edges.\n")

        # Print node breakdown for train, validation, and test sets
        train_nodes = train_mask.sum().item()
        val_nodes = valid_mask.sum().item()
        test_nodes = test_mask.sum().item()
        f.write(f"Number of train nodes: {train_nodes}\n")
        f.write(f"Number of validation nodes: {val_nodes}\n")
        f.write(f"Number of test nodes: {test_nodes}\n")

        # Basic features and labels info
        f.write(f"Features shape: {features.shape}\n")
        f.write(f"Labels shape: {labels.shape}\n")
        f.write(f"Number of labels: {num_labels}\n")
        f.write(f"Number of node types: {len(node_type_names)}\n")
        f.write(f"Number of edge types: {len(hg.edge_types)}\n")

        # Print number of distinct labels
        distinct_labels = th.sum(labels, dim=0).nonzero().size(0)
        f.write(f"Number of distinct labels: {distinct_labels}\n")

        # Print label names
        f.write("Label names:\n")
        for i, name in enumerate(label_names):
            f.write(f"  Label {i}: {name}\n")

        # Print information about each node type
        f.write("\nNode types:")
        for node_type in hg.node_types:
            f.write(f"  {node_type}: {hg[node_type].num_nodes} nodes\n")
        
        # Print information about each edge type
        f.write("\nEdge types:")
        for edge_type in hg.edge_types:
            f.write(f"  {edge_type}: {hg[edge_type].edge_index.size(1)} edges\n")

        # Print mapping of node type IDs to names
        f.write("\nNode type mapping:\n")
        for node_id, node_name in node_type_names.items():
            f.write(f"  Type {node_id}: {node_name}\n")

        # Print mapping of edge type IDs to names
        f.write("\nEdge type mapping\n")
        for edge_id, (src, relation, dst) in link_type_dic.items():
            f.write(f"  Type {edge_id}: {src} {relation} {dst}\n")
        
        # Print average number of labels per movie
        label_counts = labels.sum(dim=1)
        avg_labels = label_counts.mean().item()
        f.write(f"\nAverage number of labels per movie: {avg_labels:.2f}\n")

        in_degree, out_degree = dl.calculate_degree_stats()
        
        f.write("\nDegree Statistics:\n")
        for node_id, node_type in node_type_names.items():
            in_degrees = list(in_degree[node_id].values())
            out_degrees = list(out_degree[node_id].values())
            total_degrees = [i + o for i, o in zip(in_degrees, out_degrees)]
            
            f.write(f"\n{node_type.capitalize()}:\n")
            f.write(f"  Average in-degree: {np.mean(in_degrees):.2f}\n")
            f.write(f"  Average out-degree: {np.mean(out_degrees):.2f}\n")
            f.write(f"  Average total degree: {np.mean(total_degrees):.2f}\n")
            f.write(f"  Max in-degree: {max(in_degrees)}\n")
            f.write(f"  Max out-degree: {max(out_degrees)}\n")
            f.write(f"  Max total degree: {max(total_degrees)}\n")

        # Calculate and write total in-degree and out-degree for the entire graph
        total_in_degree = sum(sum(degrees.values()) for degrees in in_degree.values())
        total_out_degree = sum(sum(degrees.values()) for degrees in out_degree.values())
        f.write(f"\nTotal in-degree for the entire graph: {total_in_degree}\n")
        f.write(f"Total out-degree for the entire graph: {total_out_degree}\n")

        
        

def set_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans Mono'],
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'grid.color': '#E6E6E6',
        'grid.linestyle': '',
        'axes.grid': True,
        'axes.edgecolor': '#000000',  # Black axes borders
        'axes.linewidth': 1.5,  # Thicker axes
        'xtick.color': '#000000',  # Black ticks
        'ytick.color': '#000000',  # Black ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'font.weight': 'normal',  # Normal weight for text
        'axes.labelweight': 'normal',  # Normal weight for axis labels
        'axes.labelcolor': '#000000',  # Black axis labels
        'figure.dpi': 300  # High DPI for better resolution
    })

def create_horizontal_bar_chart(ax, y, x, title, xlabel, ylabel, cmap_name):
    set_style()
    
    cmap = plt.get_cmap(cmap_name)
    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.2, b=1),
        cmap(np.linspace(.5, 1, 256))
    )
    
    norm = mcolors.Normalize(vmin=min(x), vmax=max(x))
    colors = [truncated_cmap(norm(value)) for value in x]
    
    bars = ax.barh(y, x, color=colors)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, max(x) * 1.1)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01,
                bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}',  
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax.invert_yaxis()


def create_horizontal_bar_chart(ax, y, x, title, xlabel, ylabel, cmap_name):
    set_style()
    
    # Create a truncated colormap to avoid very light colors
    cmap = plt.get_cmap(cmap_name)
    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.2, b=1),
        cmap(np.linspace(.5, 1, 256))
    )
    
    # Create color gradient
    norm = mcolors.Normalize(vmin=min(x), vmax=max(x))
    colors = [truncated_cmap(norm(value)) for value in x]
    
    bars = ax.barh(y, x, color=colors)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, max(x) * 1.1)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01,
                bar.get_y() + bar.get_height()/2, 
                f'{round(width):,}',
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax.invert_yaxis()

def visualize_node_distribution(hg, node_type_names):
    node_counts = [hg[node_type].num_nodes for node_type in node_type_names.values()]
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    create_horizontal_bar_chart(ax, list(node_type_names.values()), node_counts,
                                "Distribution of Node Types", "Number of Nodes", "Node Type",
                                "Blues")
    
    plt.tight_layout()
    plt.savefig('node_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_edge_distribution(hg, link_type_dic):
    edge_counts = [hg[edge_type].edge_index.size(1) for edge_type in hg.edge_types]
    edge_labels = [f"{src} to {dst}" for src, _, dst in link_type_dic.values()]
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    create_horizontal_bar_chart(ax, edge_labels, edge_counts,
                                "Distribution of Edge Types", "Number of Edges", "Edge Type",
                                "Greens")
    
    plt.tight_layout()
    plt.savefig('edge_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_label_distribution(labels, label_names):
    label_counts = labels.sum(dim=0).numpy()
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    create_horizontal_bar_chart(ax, label_names, label_counts,
                                "Distribution of Movie Labels", "Number of Movies", "Label",
                                "Reds")
    
    plt.tight_layout()
    plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_multi_label_distribution(labels):
    label_counts = labels.sum(dim=1).numpy()
    count_distribution = Counter(label_counts)
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    create_horizontal_bar_chart(ax, list(count_distribution.keys()), list(count_distribution.values()),
                                "Distribution of Labels per Movie", "Number of Movies", "Number of Labels per Movie",
                                "Purples")
    
    plt.tight_layout()
    plt.savefig('multi_label_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_graph_sample(hg, node_type_names, link_type_dic, n_sample=500):
    set_style()
    G = nx.Graph()
    
    # Sample nodes
    sampled_nodes = {}
    node_map = {}
    for node_type in node_type_names.values():
        num_nodes = hg[node_type].num_nodes
        sample = np.random.choice(num_nodes, min(n_sample, num_nodes), replace=False)
        sampled_nodes[node_type] = sample
        node_map[node_type] = {original: i for i, original in enumerate(sample)}
    
    # Add nodes to the graph
    for node_type, nodes in sampled_nodes.items():
        G.add_nodes_from([(f"{node_type}_{i}", {"type": node_type}) for i in range(len(nodes))])
    
    # Add edges to the graph
    for (src_type, edge_type, dst_type) in hg.edge_types:
        edges = hg[src_type, edge_type, dst_type].edge_index.t().numpy()
        for i in range(edges.shape[0]):
            src, dst = edges[i]
            if src in node_map[src_type] and dst in node_map[dst_type]:
                src_idx = node_map[src_type][src]
                dst_idx = node_map[dst_type][dst]
                G.add_edge(f"{src_type}_{src_idx}", f"{dst_type}_{dst_idx}")
    
    # Set up colors for different node types
    color_map = {"movie": "#FF9999", "director": "#66B2FF", "actor": "#99FF99", "keyword": "#FFCC99"}
    node_colors = [color_map[G.nodes[node]["type"]] for node in G.nodes()]
    
    # Draw the graph
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=50, alpha=0.8, width=0.1)
    

    
    # Add a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=node_type,
                                  markerfacecolor=color, markersize=10)
                       for node_type, color in color_map.items()]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Sample of IMDB Graph Structure", fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.savefig('graph_sample.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_all_distributions(hg, node_type_names, link_type_dic, labels, label_names):
    node_counts = [hg[node_type].num_nodes for node_type in node_type_names.values()]
    edge_counts = [hg[edge_type].edge_index.size(1) for edge_type in hg.edge_types]
    edge_labels = [f"{src} to {dst}" for src, _, dst in link_type_dic.values()]
    label_counts = labels.sum(dim=0).numpy()
    label_per_movie_counts = labels.sum(dim=1).numpy()
    multi_label_distribution = Counter(label_per_movie_counts)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjusted spacing
    
    create_horizontal_bar_chart(axs[0, 0], list(node_type_names.values()), node_counts,
                                "Node Type Distribution", "Number of Nodes", "Node Types", "Blues")
    
    create_horizontal_bar_chart(axs[0, 1], edge_labels, edge_counts,
                                "Edge Type Distribution", "Number of Edges", "Edge Types", "Greens")
    
    create_horizontal_bar_chart(axs[1, 0], label_names, label_counts,
                                "Movie Label Distribution", "Number of Movies", "Labels", "Reds")
    
    create_horizontal_bar_chart(axs[1, 1], list(multi_label_distribution.keys()), 
                                list(multi_label_distribution.values()),
                                "Distribution of Labels per Movie", "Number of Movies", "Number of Labels per Movie", "Purples")
    
    # Ensure y-axis labels are visible for all subplots
    for ax in axs.flat:
        ax.tick_params(axis='y', which='both', left=True, labelleft=True)
    
    plt.tight_layout()
    plt.savefig('all_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__": 
    hg, features, labels, num_labels, train_indices, valid_indices, test_indices, \
    train_mask, valid_mask, test_mask, node_type_names, link_type_dic, label_names = load_imdb(feat_type=0, random_state=42)
    printInfo('info.txt')
    visualize_node_distribution(hg, node_type_names)
    visualize_edge_distribution(hg, link_type_dic)
    visualize_label_distribution(labels, label_names)
    visualize_multi_label_distribution(labels)
    visualize_graph_sample(hg, node_type_names, link_type_dic)
    visualize_all_distributions(hg, node_type_names, link_type_dic, labels, label_names)
