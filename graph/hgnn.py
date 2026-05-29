import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F

class GATedge(nn.Module):
    '''
    Machine node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        '''
        :param in_feats: tuple, input dimension of (operation node, machine node)
        :param out_feats: Dimension of the output (machine embedding)
        :param num_head: Number of heads
        '''
        super(GATedge, self).__init__()
        self._num_heads = num_head  # single head is used in the actual experiment
        self._in_src_feats = in_feats[0]
        self._in_dst_feats = in_feats[1]
        self._out_feats = out_feats

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_head, bias=False)
            self.fc_edge = nn.Linear(
                2, out_feats * num_head, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
        self.attn_l = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_r = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_e = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_head * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)

    def forward(self, feat, ope_ma_adj_batch, batch_idxes):
        # Two linear transformations are used for the machine nodes and operation nodes, respective
        # In linear transformation, an W^O (\in R^{d \times 7}) for \mu_{ijk} is equivalent to
        # W^{O'} (\in R^{d \times 6}) and W^E (\in R^{d \times 1}) for the nodes and edges respectively

        # Apply feature dropout
        h_src = self.feat_drop(feat[0])  # shape: (10, 5, 3)
        h_dst = self.feat_drop(feat[2])  # shape: (10, 97, 4)

        # Ensure fc_src and fc_dst are initialized properly
        if not hasattr(self, 'fc_src'):
            self.fc_src, self.fc_dst = self.fc, self.fc

        # Apply linear transformations
        feat_src = self.fc_src(h_src)  # shape: (10, 5, <output_dim>)
        feat_dst = self.fc_dst(h_dst)  # shape: (10, 97, <output_dim>)
        feat_edge = self.fc_edge(feat[1])  # shape: (10, 97, 5, <output_dim>)

        # Calculate attention coefficients
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)  # shape: (10, 5, 1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)  # shape: (10, 97, 1)
        ee = (feat_edge * self.attn_l).sum(dim=-1).unsqueeze(-1)  # shape: (10, 97, 5, 1)

        # Adjust el to match the dimensions of the edge feature
        el_unsqueezed = el.unsqueeze(1).expand(-1, ope_ma_adj_batch[batch_idxes].size(1), -1, -1)  # shape: (10, 1, 5, 1)
        el_add_ee = ope_ma_adj_batch[batch_idxes] * el_unsqueezed + ee  # shape: (10, 97, 5, 1)
        er_unsqueezed = er.unsqueeze(-1).expand(-1, -1, ope_ma_adj_batch[batch_idxes].size(2), -1)
        a = el_add_ee + ope_ma_adj_batch[batch_idxes] * er_unsqueezed  # shape: (10, 97, 5, 1)

        # Apply activation
        eijk = self.leaky_relu(a)  # shape: (10, 97, 5, 1)
        ekk = self.leaky_relu(er + er)  # shape: (10, 97, 1)

        # Normalize attention coefficients
        mask = torch.cat(
            (
                ope_ma_adj_batch[batch_idxes] == 1,  # shape: (10, 97, 5, 1)
                torch.full(
                    size=(
                        ope_ma_adj_batch[batch_idxes].size(0),
                        ope_ma_adj_batch[batch_idxes].size(1),
                        1,
                        1,
                    ),
                    dtype=torch.bool,
                    fill_value=True,
                ),
            ),
            dim=-2,
        )  # shape: (10, 97, 6, 1)

        e = torch.cat((eijk, ekk.unsqueeze(-2)), dim=-2)  # shape: (10, 97, 6, 1)
        e[~mask] = float('-inf')
        alpha = F.softmax(e.squeeze(-1), dim=-1)  # shape: (10, 97, 6)

        # Split the alpha coefficients
        alpha_ijk = alpha[..., :-1]  # shape: (10, 97, 5)
        alpha_kk = alpha[..., -1].unsqueeze(-2)  # shape: (10, 97, 1)

        # Calculate and return machine embedding
        Wmu_ijk = feat_edge + feat_src.unsqueeze(1)  # shape: (10, 97, 5, <output_dim>)
        a = Wmu_ijk * alpha_ijk.unsqueeze(-1)  # shape: (10, 97, 5, <output_dim>)
        b = torch.sum(a, dim=-2)  # shape: (10, 97, <output_dim>)
        c = feat_dst * alpha_kk.squeeze(-2).unsqueeze(-1)  # shape: (10, 97, <output_dim>)
        nu_k_prime = torch.sigmoid(b + c)  # shape: (10, 97, <output_dim>)

        if torch.any(torch.isnan(nu_k_prime)):
            print(f'help')
        return nu_k_prime


class MLPsim(nn.Module):
    '''
    Part of operation node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False):
        '''
        :param in_feats: Dimension of the input vectors of the MLPs
        :param out_feats: Dimension of the output (operation embedding) of the MLPs
        :param hidden_dim: Hidden dimensions of the MLPs
        :param num_head: Number of heads
        '''
        super(MLPsim, self).__init__()
        self._num_heads = num_head
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats),
        )

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, self._num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, feat, adj, batch_idxes):
        # MLP_{\theta_x}, where x = 1, 2, 3, 4
        # Note that message-passing should along the edge (according to the adjacency matrix)
        a = adj * feat
        b = torch.sum(a, dim=-2)

        non_zero = adj.sum(dim=2)
        # Handle division by zero by replacing it with 1
        non_zero = torch.where(non_zero == 0.0, torch.tensor(1.0), non_zero.float())
        b = b.div(non_zero)

        if torch.any(torch.isnan(b)):
            print(f'help')

        c = self.project(b)
        mask = torch.all(b == 0, dim=2)
        mask = mask.unsqueeze(2).expand(-1, -1, c.size(2))
        c[mask] = 0
        return c
