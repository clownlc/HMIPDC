from __future__ import print_function, division

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from torch.optim import Adam

from GNN import GraphAttentionLayer
from evaluation import eva
from utils import load_data, load_graph


class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(DAEGC, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GraphAttentionLayer(num_features, hidden_size, alpha)
        self.conv2 = GraphAttentionLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = dot_product_decode(z)
        return A_pred, z


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def pretrain_daegc(dataset):
    model = DAEGC(num_features=args.input_dim, hidden_size=args.hidden1_dim,
                  embedding_size=args.hidden2_dim, alpha=args.alpha).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Some porcess
    adj, adj_label = load_graph(args.name, args.k)
    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t = 2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).cuda()

    adj = adj_dense.cuda()
    adj_label = adj_label.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).cuda()
    y = dataset.y

    for epoch in range(5):
        model.train()
        A_pred, z = model(data, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, z = model(data, adj, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
            acc,nmi,ari,f1=eva(y, kmeans.labels_, epoch)
            print(epoch, acc)

        torch.save(model.state_dict(), f'predaegc_{args.name}.pkl')
    print("_____")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='dblp')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--hidden1_dim', default=1024, type=int)
    parser.add_argument('--hidden2_dim', default=128, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    # dataset = load_data(args.name)

    # 0.1 0.01
    # 3703 3703
    # 69.7 71.04
    # 1024 128 128
    # 71.6 256
    if args.name == 'cite':
        args.loss = 0.1
        args.hidden1_dim = 1024
        args.hidden2_dim = 128
        args.clu_dim = 128
        args.lr = 0.01
        args.k = None
        args.n_clusters = 6
        args.input_dim = 3703
        args.pretrain_path = 'predaegc_cite.pkl'

    # 0.1 0.01
    # 334 334
    # 76.8 79.91
    # 1024 128 128
    # 80.2 528
    elif args.name == 'dblp':
        args.loss = 0.1
        args.hidden1_dim = 1024
        args.hidden2_dim = 128
        args.clu_dim = 128
        args.lr = 0.01
        args.k = None
        args.n_clusters = 4
        args.input_dim = 334
        args.pretrain_path = 'predaegc_dblp.pkl'

    # 0.1 0.01
    # 256 30
    # 79.7
    # 0.1 2048 128 128 0.01 5 10 30
    # 78.5 90
    # 0.0 1024 64 64 0.01 5 10 100
    # 78.1 84
    elif args.name == 'usps':
        args.loss = 0.1
        args.hidden1_dim = 2048
        args.hidden2_dim = 256
        args.clu_dim = 256
        args.lr = 0.01
        args.k = 5
        args.n_clusters = 10
        args.input_dim = 30

    # 0.1 0.001
    # 561 100
    # 87.2
    # 512 32 32
    # 89.8 139
    elif args.name == 'hhar':
        args.loss = 0.1
        args.hidden1_dim = 512
        args.hidden2_dim = 32
        args.clu_dim = 32
        args.lr = 0.001
        args.k = 5
        args.n_clusters = 6
        args.input_dim = 100

    # 1 0.01
    # 2000 300
    # 79.4 77.9
    # 1024 128 32
    # 69.7 530
    elif args.name == 'reut':
        args.loss = 1
        args.hidden1_dim = 1024
        args.hidden2_dim = 128
        args.clu_dim = 32
        args.lr = 0.01
        args.k = 5
        args.n_clusters = 4
        args.input_dim = 300

    # 0.1 0.01
    # 1870
    # 91.1 92.13
    # 1024 128 128
    # 91.6 1254
    elif args.name == 'acm':
        args.loss = 0.1
        args.hidden1_dim = 1024
        args.hidden2_dim = 128
        args.clu_dim = 128
        args.lr = 0.01
        args.k = None
        args.n_clusters = 3
        args.input_dim = 1870

    # 0.1 0.01
    # 1433
    #
    # 1024 128 128
    # 91.3 295
    elif args.name == 'cora':
        args.loss = 0.1
        args.hidden1_dim = 1024
        args.hidden2_dim = 128
        args.clu_dim = 128
        args.lr = 0.01
        args.k = None
        args.n_clusters = 7
        args.input_dim = 1433

    # 0.1 0.01
    # 745 100
    # 80.07
    # 1024 128 128
    # 79.5 159
    elif args.name == 'amap':
        args.loss = 0.1
        args.hidden1_dim = 512
        args.hidden2_dim = 64
        args.clu_dim = 64
        args.lr = 0.01
        args.k = None
        args.n_clusters = 8
        args.input_dim = 100

    # 0.1 0.01
    # 500
    # 69.94
    # 1024 128
    #
    elif args.name == 'pubmed':
        args.loss = 0.1
        args.hidden1_dim = 512
        args.hidden2_dim = 64
        args.clu_dim = 64
        args.lr = 0.01
        args.k = None
        args.n_clusters = 3
        args.input_dim = 100

    # 0.1 0.01
    # 8710
    # 38.94
    # 1024 128
    #
    elif args.name == 'corafull':
        args.loss = 0.1
        args.hidden1_dim = 1024
        args.hidden2_dim = 128
        args.clu_dim = 128
        args.lr = 0.01
        args.k = None
        args.n_clusters = 70
        args.input_dim = 8710

    print(args)
    dataset = load_data(args.name, args.input_dim)
    pretrain_daegc(dataset)
