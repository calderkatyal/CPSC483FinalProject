############################################
# Loosely based on RpHGNN's hgb.py 
# https://github.com/CrawlScript/RpHGNN/blob/main/rphgnn/datasets/hgb.py
############################################

import torch as th
import torch_geometric as pyg
import numpy as np
import os
import copy
from scipy import sparse
from collections import Counter, defaultdict

class data_loader:
    def __init__(self, path):
        self.path = path
        self.nodes = self.load_nodes()
        self.links = self.load_links()
        self.labels_train = self.load_labels('label.dat')
        self.labels_test = self.load_labels('label.dat.test')

        # Process the links data
        new_data = {}
        for link_type, adj in self.links['data'].items():
            adj = adj.tocoo()
            src_type = self.links['meta'][link_type][0]
            dst_type = self.links['meta'][link_type][1]
            src_shift = self.nodes["shift"][src_type]
            dst_shift = self.nodes["shift"][dst_type]
            row, col = adj.row, adj.col
            row -= src_shift
            col -= dst_shift
            shape = [self.nodes["count"][src_type], self.nodes["count"][dst_type]]
            adj = sparse.csr_matrix((adj.data, (row, col)), shape=shape)
            new_data[link_type] = adj
        self.links['data'] = new_data

    def load_labels(self, name):
        labels = {'num_classes': 0, 'total': 0, 'count': Counter(), 'data': None, 'mask': None}
        nc = 0
        mask = np.zeros(self.nodes['total'], dtype=bool)
        data = [None for i in range(self.nodes['total'])]
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                node_id, node_name, node_type, node_label = int(th[0]), th[1], int(th[2]), list(map(int, th[3].split(',')))
                for label in node_label:
                    nc = max(nc, label + 1)
                mask[node_id] = True
                data[node_id] = node_label
                labels['count'][node_type] += 1
                labels['total'] += 1
        labels['num_classes'] = nc
        new_data = np.zeros((self.nodes['total'], labels['num_classes']), dtype=int)
        for i, x in enumerate(data):
            if x is not None:
                for j in x:
                    new_data[i, j] = 1
        labels['data'] = new_data
        labels['mask'] = mask
        return labels

    def load_links(self):
        links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list)}
        with open(os.path.join(self.path, 'link.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                if r_id not in links['meta']:
                    h_type = self.get_node_type(h_id)
                    t_type = self.get_node_type(t_id)
                    links['meta'][r_id] = (h_type, t_type)
                links['data'][r_id].append((h_id, t_id, link_weight))
                links['count'][r_id] += 1
                links['total'] += 1
        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links

    def load_nodes(self):
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        with open(os.path.join(self.path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    node_id, node_name, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_attr = list(map(float, node_attr.split(',')))
                    nodes['count'][node_type] += 1
                    nodes['attr'][node_id] = node_attr
                    nodes['total'] += 1
                elif len(th) == 3:
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    nodes['count'][node_type] += 1
                    nodes['total'] += 1
                else:
                    raise Exception("Too few information to parse!")
        shift = 0
        attr = {}
        for i in range(len(nodes['count'])):
            nodes['shift'][i] = shift
            if shift in nodes['attr']:
                mat = []
                for j in range(shift, shift + nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
        nodes['attr'] = attr
        return nodes

    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sparse.coo_matrix((data, (i, j)), shape=(self.nodes['total'], self.nodes['total'])).tocsr()
    
    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i] + self.nodes['count'][i]:
                return i
    def calculate_degree_stats(self):
        in_degree = defaultdict(lambda: defaultdict(int))
        out_degree = defaultdict(lambda: defaultdict(int))
        
        for link_type, adj in self.links['data'].items():
            src_type, dst_type = self.links['meta'][link_type]
            src_degrees = np.array(adj.sum(axis=1)).flatten()
            dst_degrees = np.array(adj.sum(axis=0)).flatten()
            
            for i, degree in enumerate(src_degrees):
                out_degree[src_type][i] += degree
            for i, degree in enumerate(dst_degrees):
                in_degree[dst_type][i] += degree
    
        return in_degree, out_degree
def load_imdb(feat_type=0, random_state=None):
    prefix = './data/IMDB'
    dl = data_loader(prefix)
    link_type_dic = {
        0: ('movie', 'to_director', 'director'),
        1: ('director', 'to_movie', 'movie'),
        2: ('movie', 'to_actor', 'actor'),
        3: ('actor', 'to_movie', 'movie'),
        4: ('movie', 'to_keyword', 'keyword'),
        5: ('keyword', 'to_movie', 'movie')
    }
    node_type_names = {0: 'movie', 1: 'director', 2: 'actor', 3: 'keyword'}
    movie_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(node_type_names[int(src_type)], link_type_dic[link_type][1], node_type_names[int(dst_type)])] = dl.links['data'][link_type].nonzero()

    # Create PyG HeteroData object
    hg = pyg.data.HeteroData()

    # Add nodes
    for node_type, count in dl.nodes['count'].items():
        hg[node_type_names[node_type]].num_nodes = count

    # Add edges
    for (src_type, edge_type, dst_type), edges in data_dic.items():
        edges_array = np.array(edges)
        hg[src_type, edge_type, dst_type].edge_index = th.from_numpy(edges_array).long()

    if feat_type == 0:
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        features = th.eye(movie_num)

    labels = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]
    labels = th.FloatTensor(labels)
    num_labels = 5
    label_names = ['Romance', 'Thriller', 'Comedy', 'Action', 'Drama'] 

    train_valid_mask = dl.labels_train['mask'][:movie_num]
    test_mask = dl.labels_test['mask'][:movie_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    val_ratio = 0.2
    np.random.seed(random_state)
    random_index = np.random.permutation(len(train_valid_indices))
    split_index = int((1.0 - val_ratio) * len(train_valid_indices))
    train_indices = np.sort(train_valid_indices[random_index[:split_index]])
    valid_indices = np.sort(train_valid_indices[random_index[split_index:]])

    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    return hg, features, labels, num_labels, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), \
           node_type_names, link_type_dic, label_names
