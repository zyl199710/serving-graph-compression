import scipy.sparse as sp
import numpy as np

import json
from torch_geometric.datasets import Planetoid
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data
import torch
from torch_geometric.data import NeighborSampler
import deeprobust.graph.utils as utils
from deeprobust.graph.data import Dpr2Pyg, Pyg2Dpr
from torch_sparse import SparseTensor
import torch_sparse


class Datasets:
    '''datasets used in GraphSAINT paper'''
    def __init__(self, dataset, **kwargs):
        dataset_str='data/'+dataset+'/'

        dataset = Planetoid(dataset_str, dataset)
        dpr_data = Pyg2Dpr(dataset)

        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
        adj_full, feat, labels = dpr_data.adj, dpr_data.features, dpr_data.labels

        self.nclass = labels.max() + 1
        self.adj_full, self.feat_full, self.labels_full = adj_full, feat, labels

        self.adj_train = adj_full[np.ix_(idx_train, idx_train)]
        self.adj_val = adj_full[np.ix_(idx_val, idx_val)]
        self.adj_test = adj_full[np.ix_(idx_test, idx_test)]

        self.feat_train = feat[idx_train]
        self.feat_val = feat[idx_val]
        self.feat_test = feat[idx_test]

        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)
        self.samplers = None

        # if dataset in ['flickr', 'reddit', 'ogbn-arxiv']:
        #     adj_full = sp.load_npz(dataset_str+'adj_full.npz')
        #     self.nnodes = adj_full.shape[0]
        #     if dataset == 'ogbn-arxiv':
        #         adj_full = adj_full + adj_full.T
        #         adj_full[adj_full > 1] = 1
        #
        #     role = json.load(open(dataset_str+'role.json','r'))
        #     idx_train = role['tr']
        #     idx_test = role['te']
        #     idx_val = role['va']
        #
        #     if 'label_rate' in kwargs:
        #         label_rate = kwargs['label_rate']
        #         if label_rate < 1:
        #             idx_train = idx_train[:int(label_rate*len(idx_train))]
        #
        #     self.adj_train = adj_full[np.ix_(idx_train, idx_train)]
        #     self.adj_val = adj_full[np.ix_(idx_val, idx_val)]
        #     self.adj_test = adj_full[np.ix_(idx_test, idx_test)]
        #
        #     feat = np.load(dataset_str+'feats.npy')
        #     # ---- normalize feat ----
        #     feat_train = feat[idx_train]
        #     scaler = StandardScaler()
        #     scaler.fit(feat_train)
        #     feat = scaler.transform(feat)
        #
        #     self.feat_train = feat[idx_train]
        #     self.feat_val = feat[idx_val]
        #     self.feat_test = feat[idx_test]
        #
        #     class_map = json.load(open(dataset_str + 'class_map.json','r'))
        #     labels = self.process_labels(class_map)
        # elif dataset == ['ogbn-products']:
        #
        #     TODO =1
        #
        #
        #
        # self.labels_train = labels[idx_train]
        # self.labels_val = labels[idx_val]
        # self.labels_test = labels[idx_test]
        #
        # # self.data_full = GraphData(adj_full, feat, labels, idx_train, idx_val, idx_test)
        # self.class_dict = None
        # self.class_dict2 = None
        #
        # self.adj_full = adj_full
        # self.feat_full = feat
        # self.labels_full = labels
        # self.idx_train = np.array(idx_train)
        # self.idx_val = np.array(idx_val)
        # self.idx_test = np.array(idx_test)
        # self.samplers = None






    def process_labels(self, class_map):
        """
        setup vertex property map for output classests
        """
        num_vertices = self.nnodes
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            self.nclass = num_classes
            class_arr = np.zeros((num_vertices, num_classes))
            for k,v in class_map.items():
                class_arr[int(k)] = v
        else:
            class_arr = np.zeros(num_vertices, dtype=np.int)
            for k, v in class_map.items():
                class_arr[int(k)] = v
            class_arr = class_arr - class_arr.min()
            self.nclass = max(class_arr) + 1
        return class_arr

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if args.nlayers == 1:
            sizes = [30]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                if args.option == 0:
                    sizes = [15, 8]
                if args.option == 1:
                    sizes = [20, 10]
                if args.option == 2:
                    sizes = [25, 10]
            else:
                sizes = [10, 5]

        if self.class_dict2 is None:
            print(sizes)
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.idx_train)
                    idx = idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    continue

                self.samplers.append(NeighborSampler(adj,
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=8, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch)
        return out




class Inductive_data:
    # transductive setting to inductive online setting

    def __init__(self, args, dpr_data):

        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
        self.idx_train_org = idx_train
        print(idx_train)
        adj, features, labels = dpr_data.adj_full, dpr_data.feat_full, dpr_data.labels_full
        self.nclass = labels.max()+1

        idx = np.arange(len(labels))
        idx_unlabel = np.setdiff1d(idx, np.concatenate([idx_train, idx_val, idx_test]))

        idx_train_un = np.concatenate([idx_train, idx_unlabel])
        idx_train_un_val = np.concatenate([idx_train, idx_unlabel, idx_val])
        idx_train_un_test = np.concatenate([idx_train, idx_unlabel, idx_test])

        if args.all_train_label == False:
            self.idx_train = np.arange(len(idx_train))
        else:
            self.idx_train = np.arange(len(idx_train_un))
        self.idx_val = np.arange(len(idx_train_un_val))[-len(idx_val):]
        self.idx_test = np.arange(len(idx_train_un_test))[-len(idx_test):]

        self.adj_train = adj[np.ix_(idx_train_un, idx_train_un)]


        adj_train = utils.sparse_mx_to_torch_sparse_tensor(adj[np.ix_(idx_train_un, idx_train_un)]) # train+val
        indices_train = adj_train._indices()
        values_train = adj_train._values()

        adj_val = utils.sparse_mx_to_torch_sparse_tensor(adj[np.ix_(idx_val, idx_train_un)]) # train+val
        indices_val = adj_val._indices()
        values_val = adj_val._values()
        indices_val[0,:] = indices_val[0,:]+len(idx_train_un)
        indices_val_T = torch.stack((indices_val[1,:], indices_val[0,:]), dim=0)

        indices = torch.cat((indices_train, indices_val, indices_val_T), dim=1)
        values = torch.cat((values_train, values_val, values_val), dim=0)
        adj_val = torch.sparse.FloatTensor(indices, values, [len(idx_train_un_val),len(idx_train_un_val)])
        self.adj_val = utils.to_scipy(adj_val)

        adj_test = utils.sparse_mx_to_torch_sparse_tensor(adj[np.ix_(idx_test, idx_train_un)]) # train+test
        indices_test = adj_test._indices()
        values_test = adj_test._values()
        indices_test[0,:] = indices_test[0,:]+len(idx_train_un)
        indices_test_T = torch.stack((indices_test[1,:], indices_test[0,:]), dim=0)

        indices = torch.cat((indices_train, indices_test, indices_test_T), dim=1)
        values = torch.cat((values_train, values_test, values_test), dim=0)
        adj_test = torch.sparse.FloatTensor(indices, values, [len(idx_train_un_test),len(idx_train_un_test)])
        self.adj_test = utils.to_scipy(adj_test)

        self.labels_train = labels[idx_train_un][self.idx_train]
        self.labels_val = labels[idx_train_un_val][self.idx_val]
        self.labels_test = labels[idx_train_un_test][self.idx_test]

        self.feat_train = features[idx_train_un]
        self.feat_val = features[idx_train_un_val]
        self.feat_test = features[idx_train_un_test] 

        self.feat_dim = self.feat_train.shape[1]
        self.class_dict = None
        self.samplers = None
        self.class_dict2 = None

        self.inductive = True
        self.labels_syn = None
        self.feat_syn = None


    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]

def feature_propagation(data, args, device):
    data.feat_train, adj_train, data.labels_train = utils.to_tensor(data.feat_train, data.adj_train, data.labels_train, device=device)
    data.feat_val, adj_val, data.labels_val = utils.to_tensor(data.feat_val, data.adj_val, data.labels_val, device=device)
    data.feat_test, adj_test, data.labels_test = utils.to_tensor(data.feat_test, data.adj_test, data.labels_test, device=device)   
    
    adj_train = adj_norm(adj_train)
    adj_val = adj_norm(adj_val)
    adj_test = adj_norm(adj_test)

    data.propagated_feats_train = feature_precalculate(data.feat_train, adj_train, args)
    data.propagated_feats_val = feature_precalculate(data.feat_val, adj_val, args)

    data.propagated_feats_train_multihop = feature_precalculate_mul(data.feat_train, adj_train, args)
    data.propagated_feats_val_multihop = feature_precalculate_mul(data.feat_val, adj_val, args)
    data.propagated_feats_test_singlehop = feature_precalculate(data.feat_test, adj_test, args)

    return data

def adj_norm(adj):
    if utils.is_sparse_tensor(adj):
        adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
    else:
        adj_norm = utils.normalize_adj_tensor(adj)
    adj = adj_norm
    adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1], value=adj._values(), sparse_sizes=adj.size()).t()
    return adj

def feature_precalculate(features, adjs, args):
    for i in range(args.nlayers):
        features = torch_sparse.matmul(adjs, features)
    return features

def feature_precalculate_mul(features, adjs, args):
    emb=features
    for i in range(args.nlayers):
        features = torch_sparse.matmul(adjs, features)
        emb=torch.cat((emb,features),dim=1)
    return emb



