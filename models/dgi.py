import torch
import torch.nn as nn
from torch.nn import Parameter
from models.dgi_pre import DGI_PRE
from execute import pretrain_dgi


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, pretrain_path='best_mnist.pkl'):
        super(DGI, self).__init__()

        self.dgi_pre = DGI_PRE(n_in, n_h, activation)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(7, n_h))  # 对权重进行初始化
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.pretrain_path = pretrain_path

    def pretrain(self, path=''):
        if path == '':
            pretrain_dgi(self.dgi_pre)
        # load pretrain weights
        # --ae.load_state_dict：将torch.load加载出来的数据加载到net中
        # --torch.load：加载训练好的模型
        self.dgi_pre.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained ae from ', path)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        ret = self.dgi_pre(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)

        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(torch.squeeze(ret).unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return ret, q

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

