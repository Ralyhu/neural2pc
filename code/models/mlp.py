import enum
import time
import random
import math
from collections import Counter
from numpy.core.fromnumeric import shape
from sklearn.cluster import KMeans
from itertools import count
from numpy.lib.function_base import flip
import scipy.sparse
from scipy.spatial import distance
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, f1_score, pairwise_distances
import torch
from torch._C import dtype
#from torch._C import dtype, float64
import torch.nn.functional as F
from torch_geometric.data.dataset import to_list
from torch_sparse import coalesce
from torch_geometric.nn import SignedConv
from torch.autograd import Variable
from utility import compute_polarity, compute_Obj, compute_stats_assignments
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, N, gamma, lamb=5,
                 bias=True, K=2):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lamb = lamb
        self.K = K
        self.N = N
        self.gamma = gamma

        self.iter = 1

        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

        self.dropout = torch.nn.Dropout(0.1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

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
        self.iter += 1
        
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        return x, x
    

    def loss_polarity(self, x, A):
        x_T = x.transpose(0,1).reshape(-1)
        union_size = torch.dot(x_T, x.reshape(-1))
        num = torch.dot(x_T, torch.mm(A, x).reshape(-1))
        
        size_s1 = torch.sum(torch.square(torch.clamp(x, min=0.0)))
        size_s2 = torch.sum(torch.square(torch.clamp(x, max=0.0)))
        min_size = torch.min(size_s1, size_s2)
        k_min_size = 2.0 * min_size
        den = k_min_size + self.gamma * (union_size - k_min_size)
        polarity_gamma = - num / den
        return polarity_gamma

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