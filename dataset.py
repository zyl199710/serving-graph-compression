import numpy as np
import deeprobust.graph.utils as utils
import torch

class Inductive_data:
    # transductive setting to inductive online setting

    def __init__(self, args, dpr_data):

        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
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