from __future__ import print_function, division


import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from sklearn.preprocessing import normalize
from torch.optim import Adam

from utils import load_data, load_graph

from models.dgi_pre import DGI_PRE
from evaluation import eva


def pretrain_dgi(args, dataset):
    model = DGI_PRE(args.n_input, args.input_dim, 'prelu')  # ----------------------------------------------------------

    model.load_state_dict(torch.load('predgi_acm.pkl'))
    model.eval()

    model.cuda()
    print("model: ", model)
    optimizer = Adam(model.parameters(), lr=args.lrpre, weight_decay=args.weight_decay)

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
    data = torch.unsqueeze(data, dim=0)
    y = dataset.y

    b_xent = nn.BCEWithLogitsLoss()

    cnt_wait = 0
    best = 1e9
    best_t = 0
    #
    # # dblp: 1000; acm: 200; amap: 1000; cite: 1000
    #
    # from tqdm import tqdm
    # pbar = tqdm(range(1000))
    #
    # for epoch in pbar:
    #     model.train()
    #     optimizer.zero_grad()
    #
    #     idx = np.random.permutation(len(y))
    #     shuf_fts = data[:, idx, :]
    #
    #     lbl_1 = torch.ones(1, len(y))
    #     lbl_2 = torch.zeros(1, len(y))
    #     lbl = (torch.stack((lbl_1, lbl_2))).squeeze(1)
    #
    #     if torch.cuda.is_available():
    #         shuf_fts = shuf_fts.cuda()
    #         lbl = lbl.cuda()
    #
    #     logits_nodes, logits_locs, logits_cs, h_1 = model(data, shuf_fts, adj, True, None, None, None)
    #
    #     loss_e = b_xent(logits_nodes, lbl)
    #     loss_i = b_xent(logits_locs, lbl)
    #     loss_j = b_xent(logits_cs, lbl)
    #
    #     loss = loss_e + 2*loss_i + 0.001*loss_j
    #
    #     # print('Loss:', loss)
    #
    #     pbar.set_description(f"loss: {loss}")  # 相当于在当前长度的基础上 +1 的操作
    #
    #     if loss < best:
    #         best = loss
    #         best_t = epoch
    #         cnt_wait = 0
    #         torch.save(model.state_dict(),  f'predgi_{args.name}.pkl')
    #     else:
    #         cnt_wait += 1
    #
    #     if cnt_wait == 50:
    #         print('Early stopping!')
    #         break
    #
    #     loss.backward()
    #     optimizer.step()

    embed, _ = model.embed(data, adj, True, None)
    embed = torch.squeeze(embed)

    # cluster parameter initiate
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)  # n_init：用不同的聚类中心初始化值运行算法的次数
    y_pred = kmeans.fit_predict(embed.data.cpu().numpy())  # 训练并直接预测
    acc, nmi, ari, f1 = eva(y, y_pred, 'pae')
    print("acc:{}, nmi:{}, ari:{}, f1:{}".format(acc, nmi, ari, f1))

    embed = embed.detach().cpu().numpy()
    return embed