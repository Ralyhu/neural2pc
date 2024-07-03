import enum
import time
import random
from collections import Counter
from itertools import count
import scipy.sparse
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, f1_score
import torch
#from torch._C import dtype, float64
import torch.nn.functional as F
from torch_geometric.data.dataset import to_list
from torch_sparse import coalesce
from utility import compute_polarity, compute_stats_assignments
from models import signed_gat
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)


class SignedGAT(torch.nn.Module):
    r"""The signed graph convolutional network model from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper.
    Internally, this module uses the
    :class:`torch_geometric.nn.conv.SignedConv` operator.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of layers.
        lamb (float, optional): Balances the contributions of the overall
            objective. (default: :obj:`5`)
        bias (bool, optional): If set to :obj:`False`, all layers will not
            learn an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, hidden_channels, num_layers, n_nodes, gamma, lamb=5,
                 bias=True, aggr="mean", discrete_regularizer=None):
        super(SignedGAT, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lamb = lamb
        self.discrete_regularizer = discrete_regularizer
        self.n_nodes = n_nodes
        self.gamma = gamma

        self.dropout = torch.nn.Dropout(0.1)

        self.conv1 = signed_gat.SignedGAT(in_channels, hidden_channels // 2,
                                first_aggr=True, aggr=aggr)
        self.conv1_bn = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2_bn = torch.nn.BatchNorm1d(hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                signed_gat.SignedGAT(hidden_channels // 2, hidden_channels // 2,
                           first_aggr=False, aggr=aggr))

        self.lin0 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1_bn.reset_parameters()
        self.conv2_bn.reset_parameters()

        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def split_edges(self, edge_index, test_ratio=0.2):
        r"""Splits the edges :obj:`edge_index` into train and test edges.

        Args:
            edge_index (LongTensor): The edge indices.
            test_ratio (float, optional): The ratio of test edges.
                (default: :obj:`0.2`)
        """
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[torch.randperm(mask.size(0))[:int(test_ratio * mask.size(0))]] = 0

        train_edge_index = edge_index[:, mask]
        test_edge_index = edge_index[:, ~mask]

        return train_edge_index, test_edge_index

    def create_spectral_features(self, pos_edge_index, neg_edge_index,
                                 num_nodes=None):
        r"""Creates :obj:`in_channels` spectral node features based on
        positive and negative edges.

        Args:
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
            num_nodes (int, optional): The number of nodes, *i.e.*
                :obj:`max_val + 1` of :attr:`pos_edge_index` and
                :attr:`neg_edge_index`. (default: :obj:`None`)
        """

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        N = edge_index.max().item() + 1 if num_nodes is None else num_nodes
        
        edge_index = edge_index.to(torch.device('cpu'))

        pos_val = torch.full((pos_edge_index.size(1), ), 2, dtype=torch.float)
        neg_val = torch.full((neg_edge_index.size(1), ), 0, dtype=torch.float)
        val = torch.cat([pos_val, neg_val], dim=0)

        row, col = edge_index
        edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        val = torch.cat([val, val], dim=0)

        edge_index, val = coalesce(edge_index, val, N, N)
        val = val - 1

        # Borrowed from:
        # https://github.com/benedekrozemberczki/SGCN/blob/master/src/utils.py
        edge_index = edge_index.detach().numpy()
        val = val.detach().numpy()
        A = scipy.sparse.coo_matrix((val, edge_index), shape=(N, N))
        svd = TruncatedSVD(n_components=self.in_channels, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        import sklearn
        sklearn.preprocessing.normalize(x, copy=False)
        return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)

    def forward(self, x, pos_edge_index, neg_edge_index):
        """Computes node embeddings :obj:`z` based on positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`.

        Args:
            x (Tensor): The input node features.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        z = self.conv1_bn(torch.relu(self.lin0(self.conv1(x, pos_edge_index, neg_edge_index))))
        for conv in self.convs:
            z = self.conv2_bn(torch.relu(self.lin0(conv(z, pos_edge_index, neg_edge_index))))
        x = self.lin1(z)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        return z, x

    def discriminate(self, z, edge_index):
        """Given node embeddings :obj:`z`, classifies the link relation
        between node pairs :obj:`edge_index` to be either positive,
        negative or non-existent.

        Args:
            x (Tensor): The input node features.
            edge_index (LongTensor): The edge indices.
        """
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

    def nll_loss(self, z, pos_edge_index, neg_edge_index):
        """Computes the discriminator loss based on node embeddings :obj:`z`,
        and positive edges :obj:`pos_edge_index` and negative nedges
        :obj:`neg_edge_index`.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1), ), 0))
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1), ), 1))
        nll_loss += F.nll_loss(
            self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1), ), 2))
        return nll_loss / 3.0

    def pos_embedding_loss(self, z, pos_edge_index):
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z, neg_edge_index):
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(self, z, pos_edge_index, neg_edge_index):
        """Computes the overall objective.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + self.lamb * (loss_1 + loss_2)

    def loss_polarity(self, x, A):
        x_T = x.transpose(0,1).reshape(-1)
        union_size = torch.dot(x_T, x.reshape(-1))
        # torch.mm support only (Sparse, Dense product), reshape is needed to convert 2d tensors to 1d in order to use dot product
        num = torch.dot(x_T, torch.mm(A, x).reshape(-1))
        size_s1 = torch.sum(torch.square(torch.clamp(x, min=0.0)))
        size_s2 = torch.sum(torch.square(torch.clamp(x, max=0.0)))
        min_size = torch.min(size_s1, size_s2)
        k_min_size = 2.0 * min_size
        den = k_min_size + self.gamma * (union_size - k_min_size)
        polarity_gamma = - num / den

        if self.discrete_regularizer:
            zero_mask = (x_T >= -0.5) & (x_T <= 0.5)
            s1_mask = x_T < -0.5
            s2_mask = x_T > 0.5
            zero_x = torch.abs(x_T)
            s1_x = torch.abs(x_T + 1)
            s2_x = torch.abs(x_T - 1)
            rho = zero_mask * zero_x + s1_mask * s1_x + s2_mask * s2_x
            regularization = torch.norm(rho, p=2)
            return polarity_gamma + self.discrete_regularizer * regularization
        else:
            return polarity_gamma

    def test(self, z, pos_edge_index, neg_edge_index):
        """Evaluates node embeddings :obj:`z` on positive and negative test
        edges by computing AUC and F1 scores.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]
        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones((pos_p.size(0))),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        auc = roc_auc_score(y, pred)
        f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0

        return auc, f1

    def test_assignments(self, x, A):
        x_np = x.numpy().reshape(-1)
        x_list = x.reshape(-1).tolist()
        # initialize the solution as empty
        solution_x = None
        solution_objective_function = float("inf")
        solution_threshold = None
        # get the thresholds from the eigenvector
        thresholds = set((np.abs(x_np) * 1000).astype(int) / 1000.0)

        k = 100
        k = len(thresholds)
        if k > len(thresholds):
            k = len(thresholds)
        thresholds = random.sample(thresholds, k)

        # compute x for all the values of the threshold
        for threshold in thresholds:
            x = np.sign(x_np, where=np.abs(x_np)>=threshold, out=np.zeros_like(x_np))

            a_dot_x = A @ x
            union_size = np.inner(x,x)

            size_s1 = np.sum(np.square(np.clip(x, a_min=0.0, a_max=1.0)))
            size_s2 = np.sum(np.square(np.clip(x, a_min = -1.0, a_max=0.0)))
            min_size = min(size_s1, size_s2)
            k_min_size = 2.0 * min_size
            den = k_min_size + self.gamma * (union_size - k_min_size)

            objective_function = - np.inner(x, a_dot_x) / den

            if objective_function < solution_objective_function:
                solution_x = x
                solution_objective_function = objective_function
                solution_threshold = threshold

        solution_x_list = solution_x.tolist()
        s1 = set([nodeid for nodeid, clusterid in enumerate(solution_x_list) if clusterid == -1])
        s2 = set([nodeid for nodeid, clusterid in enumerate(solution_x_list) if clusterid == 1])
        counts = Counter(solution_x_list)
        dist_x = compute_stats_assignments(x_list, thresholds, solution_threshold)
        return solution_objective_function, counts, s1, s2, dist_x

    def __repr__(self):
        return '{}({}, {}, num_layers={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.hidden_channels,
                                                  self.num_layers)