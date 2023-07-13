import torch
import torch.nn as nn
from torch.nn import Parameter
from layers import GCN, AvgReadout, InterDiscriminator


class DGI_PRE(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI_PRE, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = InterDiscriminator(n_h, n_in)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        logits_nodes, logits_locs, logits_cs = self.disc(c, h_1, h_2, seq1, seq2)

        return logits_nodes, logits_locs, logits_cs, h_1

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

