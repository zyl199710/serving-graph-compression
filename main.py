import sys
import argparse
from SGC import SGC
import link
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import numpy as np

from sklearn.metrics import f1_score




parser = argparse.ArgumentParser()
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--dataset', type=str, default='PubMed')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=500)# 可改
parser.add_argument('--train_iters', type=int, default=600)# 可改
parser.add_argument('--vrnum', type=int, default=60)
parser.add_argument('--k', type=int, default=2)
args = parser.parse_args()
device='cpu'

data = torch.load('processed_data/{}/data.pt'.format(args.dataset))
# model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
#             nclass=data.nclass, dropout=args.dropout,
#             nlayers=args.nlayers, with_bn=False,
#             device=device).to(device)
#
# model_parameters = list(model.parameters())
# optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
# loss = F.nll_loss
# output = model(data.feat_train).squeeze()
X = data.feat_train.detach().numpy()
X_hat = data.propagated_feats_train_multihop.detach().numpy()
# print(X)


def VNG(X, X_hat, data, args):
    def weighted_kmeans(X, weights, n_clusters, max_iterations=100):
        # Normalize the weights to sum up to 1
        weights = weights / np.sum(weights)

        # Initialize the cluster centers randomly
        kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1)
        kmeans.cluster_centers_ = X[np.random.choice(X.shape[0], size=n_clusters, replace=False)]

        # Iterate until convergence or maximum iterations reached
        for _ in range(max_iterations):
            # E-step: Assign each sample to the nearest cluster
            distances = kmeans.transform(X)
            labels = np.argmin(distances, axis=1)

            # M-step: Update cluster centers based on weighted means
            for cluster in range(n_clusters):
                cluster_indices = np.where(labels == cluster)
                cluster_weighted_sum = np.sum(X[cluster_indices] * weights[cluster_indices][:, np.newaxis], axis=0)
                cluster_weighted_mean = cluster_weighted_sum / np.sum(weights[cluster_indices])
                kmeans.cluster_centers_[cluster] = cluster_weighted_mean

        return kmeans.cluster_centers_, labels

    adj = data.adj_train.toarray()
    if data.adj_train.nnz > 0:
        sum_by_column = np.sum(adj, axis=0)
        weights = np.squeeze(sum_by_column) # 样本的权重
    else:
        weights = np.ones(len(X))
    centers, labels = weighted_kmeans(X_hat, n_clusters=args.vrnum, weights=weights)
    vectors = []
    for i in range(args.vrnum):
        indices = np.where(labels == i)[0]
        array_vector = np.zeros(len(X))
        array_vector[indices] = 1/ len(indices)
        vectors.append(array_vector)
    E = np.vstack(vectors)
    # X_vr = np.dot(E, X)
    X_vr = E @ X
    # X_hat = np.hstack([X, X])
    P = E @ X_hat
    svd1 = TruncatedSVD(n_components= min(args.vrnum, len(X[0]))-1)
    svd1.fit(P)
    U_p = svd1.transform(P)
    S_p = np.diag(svd1.singular_values_)
    S_p_k = np.linalg.inv(S_p)[0:args.k]

    V_p = svd1.components_
    Q = E @ adj @ X_hat
    QV = Q @ np.transpose(V_p)
    svd2 = TruncatedSVD(n_components= args.k)
    svd2.fit(QV)
    U_qv = svd2.transform(QV)
    A_vr = U_qv @ S_p_k @ np.transpose(U_p)
    return X_vr, A_vr, E

X_vr, A_vr, E = VNG(X, X_hat,data, args)

feat_syn_test, adj_syn_test = link.process_datasets(torch.from_numpy(E), torch.from_numpy(X_vr), torch.from_numpy(A_vr), data.feat_test, data.adj_test, data.labels_test, data.idx_test, device)
print(feat_syn_test.shape, adj_syn_test.shape)
from utils_datasets import *
adj_syn_test_norm = adj_norm(adj_syn_test)
feats_syn_singlehop = feature_precalculate(feat_syn_test, adj_syn_test_norm, args)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro

model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
            nclass=data.nclass, dropout=args.dropout,
            nlayers=args.nlayers, with_bn=False,
            device=device).to(device)

model_parameters = list(model.parameters())
optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
loss = F.nll_loss
best_acc = 0
for epoch in range(args.train_iters):
    model.train()
    optimizer_model.zero_grad()
    num_columns = data.propagated_feats_train_multihop.shape[1]
    output_train = model(data.propagated_feats_train_multihop[:, -num_columns // 3:])
    loss_train = loss(output_train[data.idx_train_org], data.labels_train[data.idx_train_org])
    loss_train.backward()
    optimizer_model.step()

    with torch.no_grad():
        model.eval()
        num_columns = data.propagated_feats_val_multihop.shape[1]
        output_val = model(data.propagated_feats_val_multihop[:, -num_columns // 3:])
        acc_val = accuracy(output_val[data.idx_val], data.labels_val)
        acc_val = acc_val.item()
    if acc_val > best_acc:
        best_acc = acc_val
        best_model = model
    print("best_acc:", best_acc, "  ", "acc_epoch:", acc_val)

# print(data.propagated_feats_test_singlehop.shape)
output_test = best_model(feats_syn_singlehop.float()).squeeze()
acc_test = accuracy(output_test[args.vrnum:], data.labels_test)
_, f1_macro_test = f1(output_test[args.vrnum:],data.labels_test)
acc_test = acc_test.item()
print("Acc:",acc_test," ", "Macro-F1:", f1_macro_test)
# SGC 调用 forward 函数 其他不用管
# 用验证集选择最好模型进行测试