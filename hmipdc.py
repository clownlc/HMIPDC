from __future__ import print_function, division

import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from torch.nn.parameter import Parameter
from torch.optim import Adam

from GNN import GraphAttentionLayer
from evaluation import eva
from abc import ABC
from utils import load_data, load_graph

from preembedding import pretrain_dgi

from utils import setup_seed


class BaseLoss:

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class DDCLoss(BaseLoss, ABC):
    """
    Michael Kampffmeyer et al. "Deep divergence-based approach to clustering"
    """

    def __init__(self, num_cluster, epsilon=1e-9, rel_sigma=0.15, device='cpu'):
        """

        :param epsilon:
        :param rel_sigma: Gaussian kernel bandwidth 高斯核带宽
        """
        super(DDCLoss, self).__init__()
        self.epsilon = epsilon
        self.rel_sigma = rel_sigma
        self.device = device
        self.num_cluster = num_cluster

    # 预测标签(logist)，低维表征(hidden)
    def __call__(self, logist, hidden):
        hidden_kernel = self._calc_hidden_kernel(hidden)

        # 考虑到簇的可分性和紧凑性
        l1_loss = self._l1_loss(logist, hidden_kernel, self.num_cluster)

        # 观测空间中的簇正交性
        l2_loss = self._l2_loss(logist)

        # 簇隶属向量接近单形的一个角（聚类成员与单纯形角的紧密性。簇分配向量的分布应该是紧密环绕在单形角的周围。）
        l3_loss = self._l3_loss(logist, hidden_kernel, self.num_cluster)

        return l1_loss + l2_loss + l3_loss

    def _l1_loss(self, logist, hidden_kernel, num_cluster):
        return self._d_cs(logist, hidden_kernel, num_cluster)

    def _l2_loss(self, logist):
        n = logist.size(0)

        # @: 正常矩阵乘法
        return 2 / (n * (n - 1)) * self._triu(logist @ torch.t(logist))

    def _l3_loss(self, logist, hidden_kernel, num_cluster):
        if not hasattr(self, 'eye'):
            self.eye = torch.eye(num_cluster, device=device)
        m = torch.exp(-self._cdist(logist, self.eye))
        return self._d_cs(m, hidden_kernel, num_cluster)

    # 相当于正则化（基于平衡类假设的替代正则化方法）
    def _triu(self, X):
        # Sum of strictly upper triangular part
        # 严格上三角部分之和
        return torch.sum(torch.triu(X, diagonal=1))

    def _calc_hidden_kernel(self, x):
        return self._kernel_from_distance_matrix(self._cdist(x, x), self.epsilon)

    def _d_cs(self, A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.
        柯西-施瓦兹散度

        :param A: Cluster assignment matrix
        :type A:  torch.Tensor
        :param K: Kernel matrix
        :type K: torch.Tensor
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :return: CS-divergence
        :rtype: torch.Tensor
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

        nom = self._atleast_epsilon(nom, eps=self.epsilon)
        dnom_squared = self._atleast_epsilon(dnom_squared, eps=self.epsilon ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * self._triu(nom / torch.sqrt(dnom_squared))
        return d

    def _atleast_epsilon(self, X, eps):
        """
        Ensure that all elements are >= `eps`.

        :param X: Input elements
        :type X: torch.Tensor
        :param eps: epsilon
        :type eps: float
        :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
        :rtype: torch.Tensor
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    def _cdist(self, X, Y):
        """
        Pairwise distance between rows of X and rows of Y.
        X行和Y行之间的成对距离（成对距离：计算两个矩阵样本之间的距离）

        :param X: First input matrix
        :type X: torch.Tensor
        :param Y: Second input matrix
        :type Y: torch.Tensor
        :return: Matrix containing pairwise distances between rows of X and rows of Y
        :rtype: torch.Tensor
        """
        xyT = X @ torch.t(Y)
        x2 = torch.sum(X ** 2, dim=1, keepdim=True)  # X ** 2：矩阵中的每个元素各自进行平方
        y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
        d = x2 - 2 * xyT + torch.t(y2)
        return d

    def _kernel_from_distance_matrix(self, dist, min_sigma):
        """
        Compute a Gaussian kernel matrix from a distance matrix.
        从距离矩阵计算高斯核矩阵。

        :param dist: Disatance matrix
        :type dist: torch.Tensor
        :param min_sigma: Minimum value for sigma. For numerical stability.
        :type min_sigma: float
        :return: Kernel matrix
        :rtype: torch.Tensor
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        # 由于浮点错误，`dist`有时可能包含负值，因此只需将其设置为零即可
        dist = F.relu(dist)
        sigma2 = self.rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(- dist / (2 * sigma2))
        return k


class DDCModule(nn.Module):

    def __init__(self, in_features, hidden_dim, num_cluster):
        super(DDCModule, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim, momentum=0.1)  # 在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的。
        )

        self.clustering_layer = nn.Sequential(
            nn.Linear(hidden_dim, num_cluster),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.hidden_layer(x)
        y = self.clustering_layer(h)
        return y, h


class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(DAEGC, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GraphAttentionLayer(num_features, hidden_size, alpha)
        self.conv2 = GraphAttentionLayer(hidden_size, embedding_size, alpha)

        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)

        # torch.nn.functional.normalize(input, p=2.0, dim=1, eps=1e-12, out=None)
        # --归一化就是要把需要处理的数据经过处理后（通过某种算法）限制在你需要的一定范围内。首先归一化是为了后面数据处理的方便，其次是保证程序运行时收敛加快。
        # --使得原来没有可比性的数据变得具有可比性，同时又保持相比较的两个数据之间的相对关系，如大小关系。
        # ----将某一个维度除以那个维度对应的范数（默认是2-范数）
        # ----dim:0表示按列操作，则每列都是除以该列下平方和的开方；1表示按行操作，则每行都是除以该行下所有元素平方和的开方

        z = F.normalize(h, p=2, dim=1)
        # z = torch.spmm(adj, z)
        s = torch.mm(z, z.t())

        s = F.softmax(s, dim=1)
        # s = sm(s)
        z_g = torch.mm(s, z)
        z = self.gamma * z_g + z
        A_pred = dot_product_decode(z)
        return A_pred, z


def sm(z):
    z = z / 10
    z = z - torch.max(z)

    z = torch.exp(z)  # 求e^zi值
    sm_z = z / torch.sum(z, dim=1)
    return sm_z


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


class Self_DAEGC(nn.Module):
    def __init__(self, args, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(Self_DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # pre_daegc
        self.pre_daegc = DAEGC(num_features, hidden_size, embedding_size, alpha)
        # self.pre_daegc.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        self.clustering_module = DDCModule(embedding_size, args.clu_dim, num_clusters)

    def forward(self, x, adj, M):
        A_pred, z = self.pre_daegc(x, adj, M)
        y, temp_h = self.clustering_module(z)

        return A_pred, z, y, temp_h


def daegc(dataset):
    model = Self_DAEGC(args, num_features=args.input_dim, hidden_size=args.hidden1_dim,
                       embedding_size=args.hidden2_dim, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Some porcess
    adj, adj_label = load_graph(args.name, args.k)
    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t = 2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    # 拓扑学上，邻居节点通过边对目标节点进行表示，GAT只考虑图注意力的1跳相邻节点(一阶)。由于图形具有复杂的结构关系，故在编码器中使用高阶邻居。
    # --通过考虑图中的t阶邻居节点，得到邻接矩阵
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t  # np.linalg.matrix_power()：矩阵的次方
    M = torch.Tensor(M_numpy).cuda()
    adj = adj_dense.cuda()
    adj_label = adj_label.cuda()
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).cuda()
    # y = dataset.y
    y = torch.tensor(dataset.y, dtype=torch.long).cuda()
    # eva(y, y_pred, 'pae')

    cls_criterion = DDCLoss(args.n_clusters, device=args.device)

    from tqdm import tqdm
    pbar = tqdm(range(2000))

    maxAcc = -1
    best_nmi = -1
    best_ari = -1
    best_f1 = -1
    best_nmi_epoch = 0
    best_ari_epoch = 0
    best_f1_epoch = 0
    best_epoch = 0

    for epoch in pbar:
        model.train()
        A_pred, z, y_pred, temp_h = model(data, adj, M)

        pred, pred_idx = y_pred.max(1)
        positive_idx = pred > args.max_positive
        negative_idx = pred < args.min_negative
        p_loss = 0
        n_loss = 0

        # nl_mask = (y_pred > args.max_positive) * 1
        nl_mask = (y_pred < args.min_negative) * 1

        # positive learning
        if sum(positive_idx * 1) > 0:
            # if args.name == 'cite' or args.name == 'hhar':
            #     p_loss += F.cross_entropy(y_pred[positive_idx], pred_idx[positive_idx] - 1, reduction='mean')
            # else:
            p_loss += F.cross_entropy(y_pred[positive_idx], pred_idx[positive_idx], reduction='mean')

        # negative learning
        if sum(negative_idx * 1) > 0:
            nl_y_pred = y_pred[negative_idx]
            # pred_nl = F.softmax(nl_y_pred, dim=1)
            pred_nl = 1 - nl_y_pred
            pred_nl = torch.clamp(pred_nl, 1e-7, 1.0)
            nl_mask = nl_mask[negative_idx]
            y_nl = torch.ones(nl_y_pred.shape).to(args.device, dtype=nl_y_pred.dtype)
            n_loss += torch.mean((-torch.sum((y_nl * torch.log(pred_nl)) * nl_mask, dim=-1))
                                 / (torch.sum(nl_mask, dim=-1) + 1e-7))

        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
        # torch.view(-1)：相当于flatten

        # clu_loss = cls_criterion(y_pred, temp_h) + cls_criterion_hidden(temp_h, z)

        clu_loss = cls_criterion(y_pred, z)
        loss = args.clu_loss * clu_loss + re_loss + args.pn_loss * (p_loss + n_loss)

        # print(clu_loss, re_loss, loss, sep='----')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc, nmi, ari, f1 = eva(y.detach().cpu().numpy(), np.argmax(y_pred.detach().cpu().numpy(), axis=1), 'pae')
        if acc > maxAcc:
            maxAcc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1
            best_epoch = epoch


        desc = epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
        ', f1 {:.4f}'.format(f1)
        # pbar.set_description("clu_loss:{} re_loss:{} acc:{} maxAcc:{} best_epoch:{}".format(clu_loss,re_loss,acc,maxAcc,best_epoch))  # 相当于在当前长度的基础上 +1 的操作
        pbar.set_description(
            "acc:{} nmi:{} ari:{} f1:{} maxAcc:{} best_nmi:{} best_ari:{} best_f1:{} best_epoch:{}".format(acc, nmi, ari, f1, maxAcc, best_nmi, best_ari, best_f1, best_epoch))  # 相当于在当前长度的基础上 +1 的操作
    return maxAcc, best_nmi, best_ari, best_f1, best_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='acm')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--i', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrpre', type=float, default=0.0001)  # ??????
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=0, type=int)  # [1,3,5]
    parser.add_argument('--max_positive', type=float, default=0.7)
    parser.add_argument('--min_negative', type=float, default=0.2)
    parser.add_argument('--hidden1_dim', default=1024, type=int)
    parser.add_argument('--hidden2_dim', default=128, type=int)
    parser.add_argument('--clu_dim', default=128, type=int)
    parser.add_argument('--input_dim', default=100, type=int)
    parser.add_argument('--clu_loss', default=1.0, type=float)
    parser.add_argument('--pn_loss', default=0.01, type=float)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    # setup_seed(args.seed)

    # 0.1 0.01
    # 3703 512
    # 69.7 71.04
    # 1024 128 128
    # 71.89 741
    if args.name == 'cite':
        # args.hidden1_dim = 1024
        # args.hidden2_dim = 128
        # args.clu_dim = 128
        # args.lr = 0.01
        # args.lrpre = 0.0001
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703
        args.input_dim = 150
        args.max_positive = 0.9
        args.min_positive = 0.3

    # 0.5 0.2 1 0.01
    # 334 50
    # 76.8 79.91
    # 1024 128 128
    # 80.6 46
    elif args.name == 'dblp':
        # args.hidden1_dim = 1024
        # args.hidden2_dim = 128
        # args.clu_dim = 128
        # args.lr = 0.01
        # args.lrpre = 0.0001
        args.k = None
        args.n_clusters = 4
        args.n_input = 334
        args.input_dim = 100
        args.max_positive = 0.7
        args.min_positive = 0.2

    # 0.1 0.01
    # 1870
    # 91.1 92.13
    # 1024 128 128
    # 91.6 1254
    elif args.name == 'acm':
        # args.hidden1_dim = 1024
        # args.hidden2_dim = 128
        # args.clu_dim = 128
        # args.lr = 0.001
        args.lrpre = 0.001
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870
        args.input_dim = 100
        args.max_positive = 0.6
        args.min_positive = 0.2

    # 0.1 0.01
    # 745 100
    # 80.07
    # 512 64 64
    # 80.3 452
    elif args.name == 'amap':
        # args.hidden1_dim = 512
        # args.hidden2_dim = 64
        # args.clu_dim = 64
        # args.lr = 0.01
        # args.lrpre = 0.001
        args.k = None
        args.n_clusters = 8
        args.n_input = 745
        args.input_dim = 150
        args.max_positive = 0.5
        args.min_positive = 0.1

    print(args)
    dataset = load_data(args.name, embed=None)
    embed = pretrain_dgi(args, dataset)

    dataset = load_data(args.name, embed)
    maxacc, best_nmi, best_ari, best_f1, bestepoch = daegc(dataset)
    print(maxacc, best_nmi, best_ari, best_f1, bestepoch, sep='--')

    # with open(f'parameter_{args.name}.txt', 'a+') as f:
    #     f.write(f"maxacc: {maxacc} -- bestepoch: {bestepoch} -- clu_loss: {args.clu_loss} -- pn_loss: {args.pn_loss} "
    #             f"-- max_positive: {args.max_positive} -- min_negative: {args.min_negative} "
    #             f" -- input_dim: {args.input_dim}")
    #     f.write('\n')
    #     f.flush()
    #
    # f.close()

    # with open(f'seed_{args.name}.txt', 'a+') as f:
    #     f.write(f"maxacc: {maxacc} -- bestepoch: {bestepoch} -- seed: {args.seed}")
    #     f.write('\n')
    #     f.flush()
    #
    # f.close()

    # with open(f'maxacc_{args.name}.txt', 'a+') as f:
    #     f.write(
    #         f"maxacc: {maxacc} -- best_nmi: {best_nmi} -- best_ari: {best_ari} -- best_f1: {best_f1} -- bestepoch: {bestepoch} -- lrpre: {args.lrpre} -- max_positive: {args.max_positive} -- min_negative: {args.min_negative} -- input_dim: {args.input_dim}")
    #     f.write('\n')
    #     f.flush()
    #
    # f.close()
