
# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import random

from dotmap import DotMap
from torch.optim import lr_scheduler
from tqdm import tqdm

from .model import SSR
import scipy.sparse as sp
from loguru import logger
import torch.nn as nn
from sklearn.metrics import confusion_matrix,classification_report
import logging

import timeit
def setup_seed():
   seed=100
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
class SGDNetTrainer(torch.nn.Module):
    def __init__(self, param):
        """
        Constructore of SGDNetTraininer

        :param param: parameter dictionary
        """
        super(SGDNetTrainer, self).__init__()
      
        self.param = param
        self.device = param.device
        
        self.in_dim = param.in_dim
        self.c = param.hyper_param.c
        
    
    def nmmatrix(self,A):
        rowsum = np.array(np.abs(A).sum(1)).astype(np.float32)
        rowsum[rowsum== 0] = 1
        r_inv = np.power(rowsum, -1).flatten()
        r_mat_inv = sp.diags(r_inv)
        
        snA = r_mat_inv @ A
        
        snA = snA.tocoo().astype(np.float32)
        return snA
    
    def get_normalized_matrices(self, edges, num_nodes):
        """
        Normalized signed adjacency matrix

        :param edges: signed edges
        :param num_nodes: number of nodes
        :return: normalized matrices
        """
        
        pi=np.where(edges[:,2]>0)
        ni=np.where(edges[:,2]<0)
        
        row_p, col_p, data_p = edges[pi, 0].squeeze(0), edges[pi, 1].squeeze(0), edges[pi, 2].squeeze(0)
        row_n, col_n, data_n = edges[ni, 0].squeeze(0), edges[ni, 1].squeeze(0), edges[ni, 2].squeeze(0)
        
        shaping = (num_nodes, num_nodes)
        A_p = sp.csr_matrix((data_p, (row_p, col_p)), shape=shaping)
        
        A_x=sp.csr_matrix((np.abs(data_n), (row_n, col_n)), shape=shaping)
       
        
        return A_p,A_x
    def convert_torch_sparse(self, A, shaping):
        """
        Convert sparse matrix into torch sparse matrix

        :param A: scipy spares matrix
        :param shaping: shape
        :return: torch sparse matrix
        """
        A = A.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((A.row, A.col)).astype(np.int64))
        values = torch.from_numpy(A.data)
    
        return torch.sparse.FloatTensor(indices, values, shaping)
    
    
    
    
    def convert_data(self, data):
        """
        Convert input data for torch
        :param data: input data
        :return: torch data
        """
        converted_data = DotMap()
        converted_data.num_nodes = data.num_nodes
        self.num_nodes=data.num_nodes
        converted_data.neg_ratio = data.neg_ratio
        converted_data.H = torch.FloatTensor(data.H).to(self.device)
      
     
        converted_data.train.edges = torch.from_numpy(data.train.edges).to(self.device)
        #print(data.train.X)
        A_p,A_n= self.get_normalized_matrices(data.train.X, data.num_nodes)
       
        A_p=self.convert_torch_sparse(A_p, A_p.shape)
        A_n=self.convert_torch_sparse(A_n, A_n.shape)


        self.Ap=A_p
        self.Am=A_n
        converted_data.train.Ap = A_p.to(self.device)
        converted_data.train.Am = A_n.to(self.device)
        
        
        
        
        y = np.asarray([1 if y_val > 0 else 0 for y_val in data.train.y])
        converted_data.train.y = torch.from_numpy(y).to(self.device)
        converted_data.class_weights = torch.from_numpy(data.class_weights).type(torch.float32).to(self.device)

        
        converted_data.test.edges = torch.from_numpy(data.test.X[:, 0:2]).to(self.device)
        y = np.asarray([1 if y_val > 0 else 0 for y_val in data.test.y])
        converted_data.test.y = torch.from_numpy(y).to(self.device)

        return converted_data

    def train_with_hyper_param(self, data, hyper_param, step,wdec,epochs=1000):
        """
        Train SGDNet with given hyperparameters
        :param data: input data
        :param hyper_param: hyperparameters
        :param epochs: target number of epochs
        :return: trained model
        """
        setup_seed()
        self.c = hyper_param.c
        converted_data = self.convert_data(data)

        model = SSR(hid_dims=hyper_param.hid_dims,
                       in_dim=hyper_param.in_dim,
                       device=self.device,
                       num_nodes=converted_data.num_nodes,
                       num_layers=hyper_param.num_layers,
                       ).to(self.device)
        print(hyper_param.learning_rate)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hyper_param.learning_rate,
                                     weight_decay=hyper_param.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=step,
                                                    gamma=wdec)

        model.train()
        
        pbar = tqdm(range(epochs), desc='Epoch...')
     
        # 1.显示创建
        logging.basicConfig(filename='logger.log', format='%(asctime)s - %(levelname)s - %(message)s',level=logging.INFO)

        # 2.定义logger,设定setLevel，FileHandler，setFormatter
        logger = logging.getLogger(__name__)   	#定义一次就可以，其他地方需要调用logger,只需要直接使用logger就行了
        logger.setLevel(level=logging.INFO)  	#定义过滤级别)
        for epoch in pbar:
            model.train()
            setup_seed()
            optimizer.zero_grad()
            loss = model(
                         Ap=converted_data.train.Ap,
                         Am=converted_data.train.Am,
                         X=converted_data.H,
                         edges=converted_data.train.edges,
                         y=converted_data.train.y,
                        neg_ratio=converted_data.neg_ratio,
                        num_nodes=converted_data.num_nodes)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            pbar.set_description('Epoch {}: {:.4} train loss'.format(epoch, loss.item()))
            
                
                    
            
            if epoch%3000==0:#(epochs-1):
                with torch.no_grad():
                    setup_seed()
                    model.eval()
                    
                    auc,f1_macro,f1_micro,f1_binary,loss = model.emb_evaluate(test_edges=converted_data.test.edges,
                                                            test_y=converted_data.test.y)
                    

                    print('final: epoch is {} auc is{:.4f} f1mac is{:.4f} f1mic is{:.4f} f1bina is{:.4f}'.format(epoch,auc,f1_macro,f1_micro,f1_binary))
                    

        pbar.close()
        
        

      
        return model,auc,f1_macro,f1_micro,f1_binary
   
