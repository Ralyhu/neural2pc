#!/usr/bin/env python
# coding:utf-8
# author: liyu
# Note: This is based on the SGCN Implementation provided by Tyler Derr.
################################################################################

from __future__ import print_function

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from torch.nn import init

from random import randint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from utility import compute_polarity, compute_stats_assignments
from torch.nn.functional import normalize
from collections import Counter
from torch_sparse import coalesce
from sklearn.decomposition import TruncatedSVD
import scipy
import random

warnings.filterwarnings("ignore")


################################################################################

class SNEA(nn.Module):
    def __init__(self, num_nodes, final_in_dim, final_out_dim, enc,
                 class_weights, lambda_structure, gamma, discrete_regularizer, hid_dims, cuda_available=False):
        super(SNEA, self).__init__()
        self.num_nodes = num_nodes
        self.enc = enc
        self.lambda_structure = lambda_structure
        self.cuda_available = cuda_available
        if class_weights is None:
            self.CrossEntLoss = nn.CrossEntropyLoss()
        else:
            self.CrossEntLoss = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights)
            )
        self.structural_distance = nn.PairwiseDistance(p=2)
        self.weight = nn.Parameter(torch.FloatTensor(final_in_dim, final_in_dim))
        self.param_src = nn.Parameter(torch.FloatTensor(2 * final_out_dim, 3))
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.param_src)
        self.act_func = F.tanh

        self.gamma = gamma
        self.discrete_regularizer = discrete_regularizer

        self.conv1_bn = torch.nn.BatchNorm1d(hid_dims)
        self.lin0 = torch.nn.Linear(hid_dims, hid_dims)
        self.lin1 = torch.nn.Linear(hid_dims, hid_dims)
        self.lin2 = torch.nn.Linear(hid_dims, 1)
        self.dropout = torch.nn.Dropout(0.1)
       
        self.lin0.reset_parameters()        
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, nodes):
        embeds_bal, embeds_unbal = self.enc(nodes)
        combined_embedding = torch.cat([embeds_bal, embeds_unbal], dim=1)
        final_embedding = self.act_func(self.weight.mm(combined_embedding.t()))
        Z = final_embedding.t()

        x = self.lin1(Z)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = torch.tanh(x)

        return Z, x

    

    def loss(self, center_nodes, adj_lists_pos, adj_lists_neg):
        max_node_index = self.num_nodes - 1
        # get the correct nodes based on this minibatch
        i_loss2 = []
        pos_no_loss2 = []
        no_neg_loss2 = []

        i_indices = []
        j_indices = []
        ys = []
        all_nodes_set = set()
        skipped_nodes = []
        for i in center_nodes:
            # if no links then we can ignore
            if (len(adj_lists_pos[i]) + len(adj_lists_neg[i])) == 0:
                skipped_nodes.append(i)
                continue
            all_nodes_set.add(i)
            for j_pos in adj_lists_pos[i]:
                i_loss2.append(i)
                pos_no_loss2.append(j_pos)
                while True:
                    temp = randint(0, max_node_index)
                    if (temp not in adj_lists_pos[i]) and (temp not in adj_lists_neg[i]):
                        break
                no_neg_loss2.append(temp)
                all_nodes_set.add(temp)

                i_indices.append(i)
                j_indices.append(j_pos)
                ys.append(0)
                all_nodes_set.add(j_pos)
            for j_neg in adj_lists_neg[i]:
                i_loss2.append(i)
                no_neg_loss2.append(j_neg)
                while True:
                    temp = randint(0, max_node_index)
                    if (temp not in adj_lists_pos[i]) and (temp not in adj_lists_neg[i]):
                        break
                pos_no_loss2.append(temp)
                all_nodes_set.add(temp)

                i_indices.append(i)
                j_indices.append(j_neg)
                ys.append(1)
                all_nodes_set.add(j_neg)

            need_samples = 2  # number of sampling of the no links pairs
            cur_samples = 0
            while cur_samples < need_samples:
                temp_samp = randint(0, max_node_index)
                if (temp_samp not in adj_lists_pos[i]) and (temp_samp not in adj_lists_neg[i]):
                    # got one we can use
                    i_indices.append(i)
                    j_indices.append(temp_samp)
                    ys.append(2)
                    all_nodes_set.add(temp_samp)
                cur_samples += 1

        all_nodes_list = list(all_nodes_set)
        all_nodes_map = {node: i for i, node in enumerate(all_nodes_list)}
        final_embedding = self.forward(all_nodes_list)

        i_indices_mapped = [all_nodes_map[i] for i in i_indices]
        j_indices_mapped = [all_nodes_map[j] for j in j_indices]
        ys = torch.LongTensor(ys)
        if self.cuda_available:
            ys = ys.cuda()

        # now that we have the mapped indices and final embeddings we can get the loss
        loss_entropy = self.CrossEntLoss(
            torch.mm(torch.cat((final_embedding[i_indices_mapped],
                                final_embedding[j_indices_mapped]), 1),
                     self.param_src),
            ys)

        i_loss2 = [all_nodes_map[i] for i in i_loss2]
        pos_no_loss2 = [all_nodes_map[i] for i in pos_no_loss2]
        no_neg_loss2 = [all_nodes_map[i] for i in no_neg_loss2]

        tensor_zeros = torch.zeros(len(i_loss2))
        if self.cuda_available:
            tensor_zeros = tensor_zeros.cuda()

        loss_structure = torch.mean(
            torch.max(
                tensor_zeros,
                self.structural_distance(final_embedding[i_loss2], final_embedding[pos_no_loss2]) ** 2
                - self.structural_distance(final_embedding[i_loss2], final_embedding[no_neg_loss2]) ** 2
            )
        )

        return loss_entropy + self.lambda_structure * loss_structure

    def test_func(self, adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg):
        all_nodes_list = list(range(self.num_nodes))
        # no map necessary for ids as we are using all nodes
        final_embedding = self.forward(all_nodes_list)
        if self.cuda_available:
            final_embedding = final_embedding.detach().cpu().numpy()
        else:
            final_embedding = final_embedding.detach().numpy()
        # training dataset
        X_train = []
        y_train = []
        X_val = []
        y_test_true = []
        for i in range(self.num_nodes):
            for j in adj_lists_pos[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_train.append(temp)
                y_train.append(1)

            for j in adj_lists_neg[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_train.append(temp)
                y_train.append(-1)

            for j in test_adj_lists_pos[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_val.append(temp)
                y_test_true.append(1)

            for j in test_adj_lists_neg[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_val.append(temp)
                y_test_true.append(-1)

        y_train = np.asarray(y_train)
        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)
        y_test_true = np.asarray(y_test_true)
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_val)

        auc = roc_auc_score(y_test_true, y_test_pred)
        f1 = f1_score(y_test_true, y_test_pred)

        return auc, f1

    # aggiunti
    def loss_polarity(self, x, A):
        x_T = x.transpose(0,1).reshape(-1)
        union_size = torch.dot(x_T, x.reshape(-1))
        # torch.mm support only (Sparse, Dense product), reshape is needed to convert 2d tensors to 1d in order to use dot product
        num = torch.dot(x_T, torch.mm(A, x).reshape(-1))
        # @ TO RESTORE
        #polarity = - num / union_size
        
        # test
        size_s1 = torch.sum(torch.square(torch.clamp(x, min=0.0)))
        size_s2 = torch.sum(torch.square(torch.clamp(x, max=0.0)))
        #print("Size S1/S2 = ", size_s1.item(), size_s2.item())
        #assert(size_s1.item() >= 0 and size_s2.item() >= 0)
        min_size = torch.min(size_s1, size_s2)
        k_min_size = 2.0 * min_size
        den = k_min_size + self.gamma * (union_size - k_min_size)
        #assert(union_size.item() >= 0)
        #print("Surplus = ", (torch.dot(x_T, x.reshape(-1)) - 2.0 * torch.min(size_s1, size_s2)).item())
        #assert((union_size - k_min_size).item() >= 0)
        #assert(den.item() > 0)
        #print("OK:", size_s1.item(), size_s2.item(), den.item())
        polarity_gamma = - num / den
        #print(polarity.item(), polarity_gamma.item(), union_size.item(), den.item())
        #assert math.isclose(polarity.item(), polarity_gamma.item(), rel_tol=1e-06)

        if self.discrete_regularizer:
            zero_mask = (x_T >= -0.5) & (x_T <= 0.5)
            s1_mask = x_T < -0.5
            s2_mask = x_T > 0.5
            zero_x = torch.abs(x_T)
            s1_x = torch.abs(x_T + 1)
            s2_x = torch.abs(x_T - 1)
            rho = zero_mask * zero_x + s1_mask * s1_x + s2_mask * s2_x
            regularization = torch.norm(rho, p=2)
            # print(regularization)
            return polarity_gamma + self.discrete_regularizer * regularization
        else:
            return polarity_gamma
    
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
        svd = TruncatedSVD(n_components=self.in_dim, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        import sklearn
        sklearn.preprocessing.normalize(x, copy=False)
        return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)

    # variant which uses scipy sparse matrix, seems faster than using tensors
    def test_assignments(self, x, A):
        x_np = x.numpy().reshape(-1)
        #start = time.time()
        x_list = x.reshape(-1).tolist()
        #assert(v >= -1.0 and v <= 1.0 for v in x_list)

        # initialize the solution as empty
        solution_x = None
        solution_objective_function = float("inf")
        solution_threshold = None
        # get the thresholds from the eigenvector
        #thresholds = {int(np.abs(element) * 1000) / 1000.0 for element in x_list} # old inefficient
        thresholds = set((np.abs(x_np) * 1000).astype(int) / 1000.0)

        k = 100
        k = len(thresholds)
        if k > len(thresholds):
            k = len(thresholds)
        thresholds = random.sample(thresholds, k)

        # compute x for all the values of the threshold
        for threshold in thresholds:
            #x = np.array([np.sign(element) if np.abs(element) >= threshold else 0.0 for element in x_list], dtype=np.float32)
            x = np.sign(x_np, where=np.abs(x_np)>=threshold, out=np.zeros_like(x_np))

            a_dot_x = A @ x
            union_size = np.inner(x,x)

            size_s1 = np.sum(np.square(np.clip(x, a_min=0.0, a_max=1.0)))
            size_s2 = np.sum(np.square(np.clip(x, a_min = -1.0, a_max=0.0)))
            #print("Size S1/S2 = ", size_s1.item(), size_s2.item())
            #assert(size_s1.item() >= 0 and size_s2.item() >= 0)
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
        #print(counts)
        #print("Threshold = " + str(solution_threshold))
        #print("Time = "+ str(time.time()-start))
        dist_x = compute_stats_assignments(x_list, thresholds, solution_threshold)
        return solution_objective_function, counts, s1, s2, dist_x