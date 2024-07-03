#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import random


import torch.nn as nn
import torch.nn.functional as F
from .utils import compute_accuracies

class Decoder(nn.Module):
    def __init__(self, emb_size,kkk):
        """
        Constructor of Decoder
        :param emb_size: embedding size
        """
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.dim=kkk
        
        self.W = nn.Parameter(torch.FloatTensor(128,64))
        torch.nn.init.xavier_normal_(self.W)
       
        self.W2 = nn.Parameter(torch.FloatTensor(64,32))
        torch.nn.init.xavier_normal_(self.W2)
        self.W3 = nn.Parameter(torch.FloatTensor(32,2))
        torch.nn.init.xavier_normal_(self.W3)
        self.W4= nn.Parameter(torch.FloatTensor(16,2))
        torch.nn.init.xavier_normal_(self.W4)
        self.act = torch.relu
        self.CrossEntLoss=nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0,1.0]))
    def forward(self, Z, edges, y):
        """
        Forward edge features from Z into binary cross entropy loss

        :param Z: final node embeddings
        :param edges: edges
        :param y: signs
        :return: binary cross entropy loss
        """
        
        src_features = Z[edges[:, 0], :]
        dst_features = Z[edges[:, 1], :]
       
        features = torch.cat((src_features, dst_features), dim=1)
       
        scores = torch.mm(features, self.W)
      
        scores = self.act(scores)

        scores = torch.mm(scores, self.W2)
        
        scores = self.act(scores)

        scores = torch.mm(scores, self.W3)
       
        scores2 = scores
      
      
       
        log_probs = F.log_softmax(scores2, dim=1)
        loss = F.nll_loss(log_probs, y)
        
        return loss

    def evaluate(self, Z, test_edges, test_y):
        """
        Predict the test edges

        :param Z: final node embeddings
        :param test_edges: test edges
        :param test_y: test signs
        :return: auc, f1 scores, and loss
       """
        src_features = Z[test_edges[:, 0], :]
        dst_features = Z[test_edges[:, 1], :]
        features = torch.cat((src_features, dst_features), dim=1)
        scores = torch.mm(features, self.W)
       
        scores = self.act(scores)
        scores = torch.mm(scores, self.W2)
       
        scores = self.act(scores)
        scores = torch.mm(scores, self.W3)
        
        scores2 = scores
       
    
       
        
        
        log_probs = F.log_softmax(scores2, dim=1)
        loss = F.nll_loss(log_probs, test_y)
        
        probs = torch.nn.functional.softmax(scores2, dim=1)
        probs = probs.cpu().detach().numpy()
        predictions = np.argmax(probs, axis=1)
       
        scores = probs[:, 1]
        y = test_y.cpu().detach().numpy()
        
        
        auc, f1_macro,f1_micro,f1_binary = compute_accuracies(y, scores, predictions)
        return auc, f1_macro,f1_micro,f1_binary, loss
