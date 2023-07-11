import numpy as np
import torch
import deeprobust.graph.utils as utils


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

def process_datasets(E, feat_syn, adj_syn, features_test, adj_test, labels_test, batch_idx, device):
    #输入：E 文章中的PI
    # feat_syn, 小图特征
    # adj_syn, 小图adj
    # features_test, 测试图的特征 前面的len(features_test)-len(labels_test)维是大图的training节点的特征，后面len(labels_test)维是需要测试的节点特征
    # adj_test, 测试图的adj 前面的len(features_test)-len(labels_test)维是大图的adj，后面len(labels_test)维是需要测试的节点adj
    # labels_test, 测试节点的label
    # batch_idx, 测试节点的index 
    idx_A = np.arange(len(features_test)-len(labels_test))#大图的training节点的个数
    a_test = adj_test[np.ix_(batch_idx,idx_A)]
    a_test = sparse_mx_to_torch_sparse_tensor(a_test).to(device)

    a_trans = [torch.sparse.mm(a_test,torch.unsqueeze(e,1).float()) for e in E]
    a_trans = torch.mean(torch.stack(a_trans),dim=0)

    size=len(a_trans)+len(adj_syn)
    adj_syn_sparse = adj_syn.to_sparse()
    indices_syn = adj_syn_sparse._indices()
    values_syn = adj_syn_sparse._values()

    a_trans_sparse = a_trans.to_sparse()
    indices_tran = a_trans_sparse._indices()
    values_tran = a_trans_sparse._values()
    indices_tran[0,:] = indices_tran[0,:]+len(adj_syn)

    indices_tran_T = torch.stack((indices_tran[1,:], indices_tran[0,:]), dim=0)

    indices = torch.cat((indices_syn, indices_tran, indices_tran_T), dim=1)
    values = torch.cat((values_syn, values_tran, values_tran), dim=0)
    adj_syn_test = torch.sparse.FloatTensor(indices, values, [size,size]).to(device)
    adj_syn_test = utils.normalize_sparse_tensor(adj_syn_test)

    #feat
    features_test = features_test[batch_idx,:].to(device)
    feat_syn_test = torch.cat((feat_syn, features_test), dim=0)
    return feat_syn_test, adj_syn_test
