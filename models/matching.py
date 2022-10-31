import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from models.qpth.qp import QPFunction
# from qpth.qp import QPFunction

import gc
import numpy as np

from util.matching_utils import kronecker, gh, reshape_edge_feature, nanorinf_replace
from util.voting_layer import Voting
from cvxpy.expressions.expression import Expression

EPS = 1e-20

class GMNet(nn.Module):
    def __init__(self):
        super(GMNet, self).__init__()
        self.voting_layer = Voting(alpha=1000)
        self.cross_graph = nn.Linear(64, 64)
        nn.init.eye_(self.cross_graph.weight)
        nn.init.constant_(self.cross_graph.bias, 0)

    def matching(self, emb1, emb2):
        # print(data1)
        # print(data2)
        G1, H1, edge_num1 = gh(emb1)
        G2, H2, edge_num2 = gh(emb2)
        G_src = torch.tensor(G1)
        G_tgt = torch.tensor(G2)
        H_src = torch.tensor(H1)
        H_tgt = torch.tensor(H2)

        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        emb1 = F.relu(self.cross_graph(emb1))
        emb2 = F.relu(self.cross_graph(emb2))

        X2 = reshape_edge_feature(emb1.transpose(1, 2).cpu(), G_src, H_src)
        Y2 = reshape_edge_feature(emb2.transpose(1, 2).cpu(), G_tgt, H_tgt)

        Me = torch.matmul(X2.transpose(1, 2), Y2).squeeze(0) / 2
        # Mp = torch.matmul(U_src.transpose(1, 2), U_tgt).squeeze(0).detach()
        Mp = torch.matmul(emb1, emb2.transpose(1, 2)).squeeze(0)
        if X2.shape[0] == 1:
            Me, Mp = Me.unsqueeze(0), Mp.unsqueeze(0)

        # print(Mp.shape)
        a1 = Me.transpose(1, 2)
        a2 = a1.reshape(Me.shape[0], Me.shape[1] * Me.shape[2])
        K_G = kronecker(G_tgt, G_src).detach()
        # print(K_G.shape)
        # K1Me = a2 * K_G
        K1Me = torch.einsum('bn, bmn->bmn', a2, K_G)
        del K_G
        del a2
        del Me
        gc.collect()
        torch.cuda.empty_cache()
        K_H = kronecker(H_tgt, H_src).detach()
        # M = torch.einsum('bnm, bmn->bnn', K1Me, K_H.transpose(1, 2))
        M = torch.matmul(K1Me, K_H.transpose(1, 2))

        # for i in range(M.shape[0]):
        #     MMMM = M[i].detach()
        #     MMMM = Expression.cast_to_const(MMMM.cpu().detach().numpy())
        #     if MMMM.is_constant():
        #         if not MMMM.is_hermitian():
        #             print(i, '   !!!!!!!!!!!!!!!!!!!')

        del K1Me
        del K_H
        gc.collect()
        torch.cuda.empty_cache()
        Mpp = torch.flatten(Mp.transpose(1, 2), 1, -1)
        # Mpp = Mp.transpose(1, 2).reshape(Mp.shape[1] * Mp.shape[2]).cuda()
        # M = M.unsqueeze(0).cuda()
        k = (Mp.shape[1] - 1) * (Mp.shape[2] - 1)
        # print(k)
        M = torch.stack([k * torch.eye(M.shape[1], M.shape[2]).cuda() - M[i].cuda() for i in range(M.shape[0])])
        # M[0] = torch.cholesky(M[0])
        # M = M.squeeze(0)

        # cvxpy/expression 516 warning
        if Mp.shape[1] > Mp.shape[2]:
            n, m, p = M.shape[0], Mp.shape[1], Mp.shape[0]
            a = torch.zeros(p, n).cuda()
            b = torch.zeros(m, n).cuda()
            for i in range(p):
                for j in range(m):
                    a[i][j * p + i ] = 1
            for i in range(m):
                b[i][i * p:(i + 1) * p] = 1

            qp = QPFunction(check_Q_spd=False)
            G = -torch.eye(n).cuda()
            h = torch.zeros(n).cuda()
            bb = torch.ones(m).cuda()
            bbb = torch.ones(p).cuda()
            hh = torch.cat((h, bbb))
            GG = torch.cat((G, a), 0)
            [M, Mpp, GG, hh, b, bb] = [x.data.double() for x in [M, Mpp, GG, hh, b, bb]]
            s = qp(M, -Mpp, GG, hh, b, bb)

            s = s.reshape(Mp.shape[1], Mp.shape[0]).unsqueeze(0)
            s = torch.relu(s) - torch.relu(s - 1)
            # s = s-s.min().detach()
            # s = torch.sigmoid((s-s.mean().detach())/torch.sqrt(s.var().detach())).permute(0,2,1)
            s = self.voting_layer(s, torch.tensor([Mp.shape[1]]), torch.tensor([Mp.shape[0]])).permute(0, 2, 1)
            # print(s)
        elif Mp.shape[1] == Mp.shape[2]:
            b, n, m, p = M.shape[0], M.shape[1], Mp.shape[1], Mp.shape[2]
            a = torch.zeros(b, m + p, n).cuda()
            for bi in range(b):
                for i in range(p):
                    for j in range(m):
                        a[bi][i][j * p + i] = 1
                for i in range(m):
                    a[bi][i + p][i * p:(i + 1) * p] = 1
            qp = QPFunction(check_Q_spd=False)
            G = torch.stack([-torch.eye(n).cuda() for i in range(M.shape[0])])
            h = torch.stack([torch.zeros(n).cuda() for i in range(M.shape[0])])
            b = torch.stack([torch.ones(m + p).cuda() for i in range(M.shape[0])])

            [M, Mpp, G, h, a, b] = [x.data.double() for x in [M, Mpp, G, h, a, b]]
            s = qp(M, -Mpp, G, h, a, b)

            s = s.reshape(Mp.shape[0], Mp.shape[2], Mp.shape[1]).transpose(1, 2)
            s = torch.relu(s) - torch.relu(s - 1)
            # s = s-s.min().detach()
            # s = torch.sigmoid((s-s.mean().detach())/torch.sqrt(s.var().detach()))
            s = self.voting_layer(s, torch.tensor([Mp.shape[0]]), torch.tensor([Mp.shape[1]]))

        else:
            n, m, p = M.shape[0], Mp.shape[0], Mp.shape[1]
            a = torch.zeros(p, n).cuda()
            b = torch.zeros(m, n).cuda()
            for i in range(p):
                for j in range(m):
                    a[i][j * p + i] = 1
            for i in range(m):
                b[i][i * p:(i + 1) * p] = 1
            qp = QPFunction(check_Q_spd=False)
            G = -torch.eye(n).cuda()
            h = torch.zeros(n).cuda()
            bb = torch.ones(m).cuda()
            bbb = torch.ones(p).cuda()
            hh = torch.cat((h, bbb))
            GG = torch.cat((G, a), 0)
            s = qp(M, -Mpp, GG, hh, b, bb)
            s = s.reshape(Mp.shape[1], Mp.shape[0]).t().unsqueeze(0)
            s = torch.relu(s) - torch.relu(s - 1)
            # s = s-s.min().detach()
            # s = torch.sigmoid((s-s.mean().detach())/torch.sqrt(s.var().detach()))
            s = self.voting_layer(s, torch.tensor([Mp.shape[0]]), torch.tensor([Mp.shape[1]]))

        return s.unsqueeze(0)

    def forward(self, outputs, targets):
        """
        Performs the sequence level matching
        """
        bs, num_queries = outputs["pred_ins"].shape[:2]
        indices = []
        num_out = 10
        num_frame = 24
        num_ins = targets[0]["ins_ann"].shape[0] // num_frame

        for i in range(bs):
            outputs_ins = outputs["pred_ins"][i]
            targets_ins = targets[i]["ins_ann"]
            targets_ids = targets[i]["labels"]
            targets_valid = targets[i]["valid"]

            num_tgt = len(targets_ids) // num_frame
            tgt_valid_split = targets_valid.reshape(num_tgt, num_frame)

            out_ins = outputs_ins.view(num_frame, num_out, outputs_ins.shape[-1]).permute(1, 0, 2)
            # out_ins = out_ins.view(num_out, num_frame, num_ins, out_ins.shape[-1]).permute(0, 2, 1, 3)
            tgt_ins = targets_ins.view(num_frame, num_ins, targets_ins.shape[-1]).permute(1, 0, 2)

            # cost = [[torch.diagonal(self.matching(out_ins[row].unsqueeze(0),
            #                                       tgt_ins[col].unsqueeze(0)).squeeze()).sum()
            #          for col in range(tgt_ins.shape[0])] for row in range(out_ins.shape[0])]
            cost = [[0 for col in range(tgt_ins.shape[0])] for row in range(out_ins.shape[0])]
            match_gt = list()
            for i in range(out_ins.shape[0]):
                for j in range(tgt_ins.shape[0]):
                    cost_ij = self.matching(out_ins[i].unsqueeze(0),
                                               tgt_ins[j].unsqueeze(0))
                    match_gt.append(cost_ij.squeeze(1))
                    cost[i][j] = torch.diagonal(cost_ij.squeeze()).sum()
            # s = [self.matching(out_ins_, tgt_ins) for out_ins_ in out_ins]
            # cost = torch.diagonal(torch.cat(s, dim=0), dim1=2, dim2=3).sum(2)
            match_gt = torch.cat(match_gt).view(num_frame*num_ins*num_out, num_frame)
            cost = torch.Tensor(cost)
            out_i, tgt_i = linear_sum_assignment(cost.cpu())
            index_i, index_j = [], []
            for j in range(len(out_i)):
                tgt_valid_ind_j = tgt_valid_split[j].nonzero(as_tuple=False).flatten()
                index_i.append(tgt_valid_ind_j * num_out + out_i[j])
                index_j.append(tgt_valid_ind_j + tgt_i[j] * num_frame)

            if index_i == [] or index_j == []:
                indices.append((torch.tensor([]).long().to(outputs_ins.device),
                                torch.tensor([]).long().to(outputs_ins.device)))
            else:
                index_i = torch.cat(index_i).long()
                index_j = torch.cat(index_j).long()
                indices.append((index_i, index_j))

        return indices, match_gt


if __name__ == '__main__':
    data1 = F.relu(torch.Tensor(72, 64,64).to('cuda'))
    data2 = F.relu(torch.Tensor(36, 64,64).to('cuda'))

    gmm = GMNet().to('cuda')

    s = gmm(data1, data2)

    print(s)