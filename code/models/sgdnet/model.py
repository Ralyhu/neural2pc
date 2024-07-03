#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import random
from utility import compute_polarity, compute_stats_assignments
from torch.nn.functional import normalize
from collections import Counter
from torch_sparse import coalesce
from sklearn.decomposition import TruncatedSVD
import scipy
from torch.nn import Parameter
import torch.nn as nn
#from sgdnet.decoder import Decoder
from models.sgdnet.decoder import Decoder
import math
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix,classification_report
from random import randint

import timeit
#import numba


def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
   
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class SSR(torch.nn.Module):
    def __init__(self,
                 hid_dims,
                 in_dim,
                 device,
                 num_nodes,
                 num_layers=3,
                 gamma=1.0,
                 discrete_regularizer=None,
                 K=2
                 ):
        """
        Constructor of SGDNet

        :param hid_dims: hidden dimension
        :param in_dim: input dimension
        :param device: gpu device
        :param num_nodes: number of nodes (n)
        :param num_layers: number of layers (L)
        :param num_diff_layers: number of diffusion steps (K)
        :param c: ratio of local feature injection
        """
        super(SSR, self).__init__()
       
        self.hid_dims = hid_dims
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.K = K
        self.device = device
        self.num_nodes = num_nodes
        self.act = torch.tanh
        self.CrossEntLoss = nn.CrossEntropyLoss()

        self.gamma = gamma
        self.discrete_regularizer = discrete_regularizer

        self.conv1_bn = torch.nn.BatchNorm1d(hid_dims)
        self.lin0 = torch.nn.Linear(hid_dims, hid_dims)
        self.lin1 = torch.nn.Linear(hid_dims, hid_dims)
        self.lin2 = torch.nn.Linear(hid_dims, 1)
        self.dropout = torch.nn.Dropout(0.1)
       
        self.setup_layers()

    def setup_layers(self):
        """
        Set up layers
        :return: none
        """
      
        kkk=self.K
        dim=16
        self.Global1=GlobalLayer(device=self.device,kkk=kkk)
        self.Global2=GlobalLayer(device=self.device,kkk=kkk)
        self.Global3=GlobalLayer(device=self.device,kkk=kkk)
        self.Global4=GlobalLayer(device=self.device,kkk=kkk)
        self.Global5=GlobalLayer(device=self.device,kkk=kkk)

        bb=False
        self.locala=LocalLayer(device=self.device,in_channels=self.in_dim, out_channels=64-dim, norm=bb, norm_embed=bb, bias=True)
        self.localb=LocalLayer(device=self.device,in_channels=64-dim, out_channels=64-dim, norm=bb, norm_embed=bb, bias=True)
        
        

        self.color=nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim // 2),
            nn.ReLU(),
            nn.Linear(self.in_dim // 2,dim)
        )
        
        self.decoder = Decoder(64,kkk)
        self.Cx=Parameter(torch.FloatTensor(dim,kkk))
        torch.nn.init.xavier_normal_(self.Cx)
        self.Wx=Parameter(torch.FloatTensor(64*2,1)) # seems not used
        torch.nn.init.xavier_normal_(self.Wx)
        self.W2=Parameter(torch.FloatTensor(32*2,2)) # seems not used
        torch.nn.init.xavier_normal_(self.W2)
        
        self.Trans=nn.Sequential(
            nn.Linear(kkk,32,bias=False)
        )
        self.lin0.reset_parameters()        
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def l2_norm(self,input, axit=1):
        norm = torch.norm(input,2,axit,True)
        output = torch.div(input, norm)
        return output
    #def forward(self,Ap,Am,X, edges, y,neg_ratio,num_nodes):
    def forward(self,Ap,Am, X):
        """
        Forward X into loss

        :param nApT: transposed normalized matrix for + sign
        :param nAmT: transposed normalized matrix for - sign
        :param X: input node features
        :param edges: edges
        :param y: signs
        :return: BCE loss
        """
        cc=self.color(X)

        CX=torch.mm(cc,self.Cx)
        ck=F.softmax(CX,dim=1)
        
        X1=self.Global1(Ap,Am,ck)
        X2=self.Global2(Ap,Am,X1)
        X3=self.Global3(Ap,Am,X2)
        X4=self.Global4(Ap,Am,X3)
        X5=self.Global5(Ap,Am,X4)

        L1=self.locala(Ap,Am,X)
        L1=self.act(L1)
        L2=self.localb(Ap,Am,L1)

        self.XL=L2
        self.XF=X5
        XG=(torch.mm(self.Cx,self.XF.t())).t()
        self.Z=self.conv1_bn(torch.relu(self.lin0(torch.cat((XG,self.XL),dim=1))))
        self.Ap=Ap
        self.Am=Am

        #loss = self.decoder(self.Z, edges, y)
        #loss = self.loss_polarity(self.Z, A)

        # final mlp layer
        x = self.lin1(self.Z)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        return self.Z, x
    

    def emb_evaluate(self,test_edges, test_y):
        
        return self.decoder.evaluate(self.Z, test_edges, test_y)

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
    
    # experimental version
    def test_assignments_conditionalexpectation(self, x, A):
        #start = time.time()
        x_list = x.reshape(-1).tolist()
        #assert(v >= -1.0 and v <= 1.0 for v in x_list)
        x_nodeass = list(enumerate(x_list))
        x_sorted = sorted(x_nodeass, key=lambda x: abs(x[1]), reverse=True)

        x_np = np.array(x_list)
        thresholds = set((np.abs(x_np) * 1000).astype(int) / 1000.0)

        k = 100
        k = len(thresholds)
        if k > len(thresholds):
            k = len(thresholds)
        thresholds = random.sample(thresholds, k)

        solution_objective_function = float("inf")
        #solution_x = np.zeros_like(x_np)
        solution_x = np.array(x_np)
        for (i, xi) in x_sorted:
            x = np.array(solution_x)
            x[i] = 1.0 if xi > 0 else -1.0
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
                
        
        solution_threshold = 0.5 # a caso per retrocompatibilità

        solution_x_list = solution_x.tolist()
        s1 = set([nodeid for nodeid, clusterid in enumerate(solution_x_list) if clusterid == -1])
        s2 = set([nodeid for nodeid, clusterid in enumerate(solution_x_list) if clusterid == 1])
        counts = Counter(solution_x_list)
        
        dist_x = compute_stats_assignments(x_list, thresholds, solution_threshold)
        return solution_objective_function, counts, s1, s2, dist_x
        
class LocalLayer(torch.nn.Module):
    def __init__(self,
                device,
                 in_channels,
                 out_channels,
                 norm=True,
                 norm_embed=True,
                 bias=True):
        super(LocalLayer, self).__init__()
        
        self.device=device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.norm_embed = norm_embed
        self.trans=nn.Sequential(
            nn.Linear(self.in_channels*3, out_channels,bias=True)

        )
        
     
    def forward(self, nApT, nAmT, X):
        
        Xp=torch.sparse.mm(nApT, X)
        Xn=torch.sparse.mm(nAmT, X)
        out_p=torch.cat((X,Xp,Xn),1)
        out_P=self.trans(out_p)
        return out_P


class GlobalLayer(torch.nn.Module):
    def __init__(self,
                device,kkk):
        super(GlobalLayer, self).__init__()
        
        
       
        
        self.trans=nn.Sequential(
            nn.Linear(kkk*3,16),
            nn.Tanh(),
            nn.Linear(16,kkk),
            
        )
    def forward(self, nApT, nAmT, X):
        
        X1=X
        
        X2=X
        Xp=torch.sparse.mm(nApT, X1)
        Xn=torch.sparse.mm(nAmT, X2)
        XX=torch.cat((X,Xp,Xn),dim=1)
      
        C=self.trans(XX)

        colorN=torch.softmax(C,dim=1)
        return colorN
   