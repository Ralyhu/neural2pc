from collections import Counter
from math import e
import os.path as osp
import os
import sys
import argparse
import time
import traceback
import random
import numpy as np
import pandas as pd
import torch
import scipy.sparse 
#from sgdnet.model import SSR
from models.sgdnet.model import SSR
from utility import check_result_KCG
from utility import get_edges_clusters

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
basePath = os.path.dirname(os.path.abspath(__file__)) + "/../" 

basePath_out = basePath + "/output/"
basePath_data = basePath + "/datasets/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"

def convert_torch_sparse(A, shaping):
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

def train(model, optimizer, x, A, train_pos_edge_index, train_neg_edge_index, A_p, A_n):
    model.train()
    optimizer.zero_grad()
    z, assignments = model(A_p, A_n, x)
    #loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
    loss = model.loss_polarity(assignments, A)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, x, A_M, train_pos_edge_index, train_neg_edge_index, A_p, A_n):
    model.eval()
    with torch.no_grad():
        z, assignments = model(A_p, A_n, x)
    return model.test_assignments(assignments, A_M), z

if __name__ == '__main__':
    # create a parser
    parser = argparse.ArgumentParser(description='Neural2PC with SGDNET as SGNN model')

    parser.add_argument('-d', help='dataset_name', type=str, required=True)
    parser.add_argument('-e', help='Number of epochs', type=int, default=200)
    parser.add_argument('-s', help='Random seed', type=int, default=100)
    parser.add_argument('-i', help='Iteration number', type=int, default=0)
    parser.add_argument('-p', help='Percentage of last epochs to consider for average', type=int, default=10)
    parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
    parser.add_argument('-wd', help='Weight Decay (L2 regularization)', type=float, default=0.01)
    parser.add_argument('-gamma', help='Gamma weight for polarity', type=float, default=1.0)
    parser.add_argument('-dr', help='Discrete regularizer', type=bool, default=False)
    parser.add_argument('-l', help='Number of layers', type=int, default=2)
    parser.add_argument('-K', help='Number of latent communities', type=int, default=2)
    parser.add_argument('-agg', help='Aggregation operator', type=str, choices=["add", "mean", "max"], default="mean")

    args = parser.parse_args()

    dataset_name = args.d
    epochs = args.e
    it = args.i
    p = args.p
    lr = args.lr
    wd = args.wd
    dr = args.dr
    l = args.l
    agg = args.agg
    gamma = args.gamma
    K = args.K

    seed = args.s * it
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    gamma_str_path = "_gamma" + str(gamma) if gamma != 1.0 else ""
    
    if dr:
        output_path = basePath_out + dataset_name + "/sgdnet-dr_K" + str(K) + "_e" + str(epochs) + "_lr" + str(lr) + "_wd" + str(wd) + gamma_str_path + "/i_" + str(it) + "/" 
    else:
        output_path = basePath_out + dataset_name + "/sgdnet_K" + str(K) + "_e" + str(epochs) + "_lr" + str(lr) + "_wd" + str(wd) + gamma_str_path +  "/i_" + str(it) + "/" 
    # write solution to file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    try:
        pos_edge_indices_i, pos_edge_indices_j, neg_edge_indices_i, neg_edge_indices_j  = [], [], [], []
        edges = []
        # read the graph from file
        dataset_path = basePath_data + dataset_name + ".txt"
        with open(dataset_path) as f:
            n = int(f.readline().replace('# ', ''))
            print("#nodes = " + str(n))
            for line in f.readlines():
                split_line = line.split('\t')
                i = int(split_line[0])
                j = int(split_line[1])
                sign = int(split_line[2])
                edges.append((i, j, sign))
                if sign > 0:
                    pos_edge_indices_i.append(i)
                    pos_edge_indices_j.append(j)
                    
                    pos_edge_indices_i.append(j)
                    pos_edge_indices_j.append(i)
                    
                else:
                    #sign == -1
                    assert sign == -1
                    neg_edge_indices_i.append(i)
                    neg_edge_indices_j.append(j)

                    neg_edge_indices_i.append(j)
                    neg_edge_indices_j.append(i)
                    

        pos_edge_index = torch.LongTensor([pos_edge_indices_i, pos_edge_indices_j]).to(device)
        neg_edge_index = torch.LongTensor([neg_edge_indices_i, neg_edge_indices_j]).to(device)


        # build signed adjacency matrix
        ones = [1.0] * pos_edge_index.shape[1]
        A_plus_M = scipy.sparse.coo_matrix((ones, pos_edge_index.tolist()), shape=(n, n)).tocsr()
        A_plus = torch.sparse_coo_tensor(pos_edge_index.tolist(), ones, (n, n))
        minus_ones = [-1.0] * neg_edge_index.shape[1]
        A_minus_M = scipy.sparse.coo_matrix((minus_ones, neg_edge_index.tolist()), shape=(n, n)).tocsr()
        A_minus = torch.sparse_coo_tensor(neg_edge_index.tolist(), minus_ones, (n, n))
        A = A_plus + A_minus
        A_M = A_plus_M.tocsr() + A_minus_M.tocsr()

        # Build and train model.
        in_size = 64
        if n < in_size:
            in_size = n // 2 # if in_size > n then the model raise an error (e.g. cloister e highlandtribes)
        hidden_size = 64 # 32 for the "friends" and 32 the "enemies"

        if dr:
            model = SSR(in_dim=in_size, hid_dims=hidden_size, device=device, num_layers=l, num_nodes=n, gamma=gamma, discrete_regularizer=wd, K=K).to(device)
        else:
            model = SSR(in_dim=in_size, hid_dims=hidden_size, device=device, num_layers=l, num_nodes=n, gamma=gamma, discrete_regularizer=None, K=K).to(device)

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        if dr:
            wd = 0.0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        #train_pos_edge_index, test_pos_edge_index = model.split_edges(pos_edge_index)
        train_pos_edge_index = pos_edge_index
        test_pos_edge_index = pos_edge_index
        #train_neg_edge_index, test_neg_edge_index = model.split_edges(neg_edge_index)
        train_neg_edge_index = neg_edge_index
        test_neg_edge_index = neg_edge_index

        x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index, n)
        #x = A

        # lines added to accomate data format of sgdnet
        edges = np.array(edges)
        pi=np.where(edges[:,2]>0)
        ni=np.where(edges[:,2]<0)
        row_p, col_p, data_p = edges[pi, 0].squeeze(0), edges[pi, 1].squeeze(0), edges[pi, 2].squeeze(0)
        row_n, col_n, data_n = edges[ni, 0].squeeze(0), edges[ni, 1].squeeze(0), edges[ni, 2].squeeze(0)
        shaping = (n, n)
        A_p = scipy.sparse.csr_matrix((data_p, (row_p, col_p)), shape=shaping)
        A_n = scipy.sparse.csr_matrix((np.abs(data_n), (row_n, col_n)), shape=shaping)
        A_p = convert_torch_sparse(A_p, A_p.shape)
        A_n = convert_torch_sparse(A_n, A_n.shape)
        A_p = A_p.to(device)
        A_n = A_n.to(device)
        #print(A_p)
        #print(A_n)
        # -----------------------------------------------

        best_s1 = None
        best_s2 = None
        best_polarity = float("+inf")
        best_epoch = None
        best_counts = None
        best_embeddings = None
        training_time = 0.0
        test_time = 0.0
        train_loss_list = []
        test_loss_list = []
        last_polarity = []
        last_size_s1 = []
        last_size_s2 = []
        last_ag_ratio = []
        last_int_p1 = []
        last_int_p2 = []
        last_int_n1 = []
        last_int_n2 = [] 
        last_inter_p = []
        last_inter_n = []
        pos_thresholds = [] 
        neg_thresholds = []
        all_thresholds = []
        for epoch in range(epochs):
            start_t = time.time()
            loss = train(model, optimizer, x, A, train_pos_edge_index, train_neg_edge_index, A_p, A_n)
            training_time += (time.time() - start_t)
            train_loss_list.append(-loss)
            #auc, f1 = test()
            start_t = time.time()
            (polarity, counts, s1, s2, dict_x), z = test(model, x, A_M, train_pos_edge_index, train_neg_edge_index, A_p, A_n)
            test_time += (time.time() - start_t)
            test_loss_list.append(-polarity)
            pos_tuple = (dict_x["avg_pos"], dict_x["n_pos"], dict_x["min_pos"], dict_x["max_pos"])
            pos_thresholds.append(pos_tuple)
            neg_tuple = (dict_x["avg_neg"], dict_x["n_neg"], dict_x["min_neg"], dict_x["max_neg"])
            neg_thresholds.append(neg_tuple)
            all_tuple = (dict_x["avg_ths"], dict_x["n_ths"], dict_x["min_ths"], dict_x["max_ths"], dict_x["best_th"])
            all_thresholds.append(all_tuple)
            if polarity <= best_polarity:
                best_polarity = polarity
                best_s1 = s1
                best_s2 = s2
                best_epoch = epoch
                best_counts = counts
                best_embeddings = z
            if epoch >= ((100.0 - p)/100.0) * epochs:
                # convert to membership array
                membership = [-1] * n
                if len(s2) > len(s1): # associate label 1 to the biggest group, 2 to the smallest
                    tmp = s1
                    s1 = s2
                    s2 = tmp
                for v in s1:
                    membership[v] = 1
                for v in s2:
                    membership[v] = 2
                
                last_polarity.append(-polarity)
                last_size_s1.append(len(s1))
                last_size_s2.append(len(s2))
                int_p1, int_p2, int_n1, int_n2, inter_p, inter_n = get_edges_clusters(A_M, membership)
                last_int_p1.append(int_p1)
                last_int_p2.append(int_p2)
                last_int_n1.append(int_n1)
                last_int_n2.append(int_n2)
                last_inter_p.append(inter_p)
                last_inter_n.append(inter_n)
                m = int_p1 + int_p2 + int_n1 + int_n2 + inter_p + inter_n
                if m!= 0:
                    ag_ratio = ((int_p1) + (int_p2) + (inter_n))/m
                else: 
                    ag_ratio = 0.0
                last_ag_ratio.append(ag_ratio)
            #print('Epoch: {:03d}, Loss: {:.4f}, AUC: {:.4f}, F1: {:.4f}'.format(
            #    epoch, loss, auc, f1))
            print('Epoch: {:03d}, Loss: {:.4f}, counts: {}, polarity: {:.4f}'.format(epoch, loss, str(counts), -polarity))


        # convert to membership array
        membership = [-1] * n
        if len(best_s2) > len(best_s1): # associate label 1 to the biggest group, 2 to the smallest
            tmp = best_s1
            best_s1 = best_s2
            best_s2 = tmp
        for v in best_s1:
            membership[v] = 1
        for v in best_s2:
            membership[v] = 2

        print("Printing best solution found during the learning phase...")
        print("Found at epoch = " + str(best_epoch))
        print("Polarity = " + str(-best_polarity))
        print("Size of clusters : " + str(best_counts))
        print("Set s1:")
        print(best_s1)
        print("Set s2:")
        print(best_s2)
        print("Training time = " + str(training_time))
        print("Test time = "+ str(test_time))

        # write solution to file
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        membership_file = output_path + "membership.txt"
        with open(membership_file, "w+") as f:
            for c in membership:
                f.write(str(c) + "\n")

        # write the size of the discovered groups
        sizes_file = output_path + "size.txt"
        with open(sizes_file, "w+") as f:
            f.write("s1 : " + str(len(best_s1)) + "\n")
            f.write("s2 : " + str(len(best_s2)) + "\n")

        # write the polarity score
        polarity_file = output_path + "polarity.txt"
        with open(polarity_file, "w+") as f:
            f.write(str(-best_polarity))

        # write edges information
        int_p1, int_p2, int_n1, int_n2, inter_p, inter_n = get_edges_clusters(A_M, membership)
        edges_count_file = output_path + "edges_info.txt"
        with open(edges_count_file, "w+") as f:
            f.write("int_p1 : " + str(int_p1) + "\n")
            f.write("int_p2 : " + str(int_p2) + "\n")
            f.write("int_n1 : " + str(int_n1) + "\n")
            f.write("int_n2 : " + str(int_n2) + "\n")
            f.write("inter_p : " + str(inter_p) + "\n")
            f.write("inter_n : " + str(inter_n) + "\n")

        #write training time
        time_file = output_path + "training_time.txt"
        with open(time_file, "w+") as f:
            f.write(str(training_time))
        
        #write test time
        time_file = output_path + "test_time.txt"
        with open(time_file, "w+") as f:
            f.write(str(test_time))

        #write best epoch
        epoch_file = output_path + "best_epoch.txt"
        with open(epoch_file, "w+") as f:
            f.write(str(best_epoch))

        training_loss_file = output_path + "training_loss.txt"
        with open(training_loss_file, "w+") as f:
            for l in train_loss_list:
                f.write(str(round(l,3)) + "\n")

        test_loss_file = output_path + "test_loss.txt"
        with open(test_loss_file, "w+") as f:
            for l in test_loss_list:
                f.write(str(round(l,3)) + "\n")

        thresholds_file = output_path + "thresholds.txt"
        with open(thresholds_file, "w+") as f:
            for t in all_thresholds:
                line = str(round(t[0], 5)) + "," + str(t[1]) + "," + str(round(t[2], 5)) + "," + str(round(t[3], 5)) + "," + str(round(t[4], 5))
                f.write(line + "\n")

        thresholds_file = output_path + "pos_thresholds.txt"
        with open(thresholds_file, "w+") as f:
            for t in pos_thresholds:
                line = str(round(t[0], 5)) + "," + str(t[1]) + "," + str(round(t[2], 5)) + "," + str(round(t[3], 5))
                f.write(line + "\n")
        
        thresholds_file = output_path + "neg_thresholds.txt"
        with open(thresholds_file, "w+") as f:
            for t in neg_thresholds:
                line = str(round(t[0], 5)) + "," + str(t[1]) + "," + str(round(t[2], 5)) + "," + str(round(t[3], 5))
                f.write(line + "\n")

        # ----------------------------
        # write info about last epochs
        # ----------------------------

        # write the size of the discovered groups
        sizes_file = output_path + "last_size.txt"
        with open(sizes_file, "w+") as f:
            f.write("s1 : " + str(sum(last_size_s1)/len(last_size_s1)) + "\n")
            f.write("s2 : " + str(sum(last_size_s2)/len(last_size_s2)) + "\n")

        # write the polarity score
        polarity_file = output_path + "last_polarity.txt"
        with open(polarity_file, "w+") as f:
            f.write(str(sum(last_polarity)/len(last_polarity)))

        # write edges information
        edges_count_file = output_path + "last_edges_info.txt"
        with open(edges_count_file, "w+") as f:
            f.write("int_p1 : " + str(sum(last_int_p1)/len(last_int_p1)) + "\n")
            f.write("int_p2 : " + str(sum(last_int_p2)/len(last_int_p2)) + "\n")
            f.write("int_n1 : " + str(sum(last_int_n1)/len(last_int_n1)) + "\n")
            f.write("int_n2 : " + str(sum(last_int_n2)/len(last_int_n2)) + "\n")
            f.write("inter_p : " + str(sum(last_inter_p)/len(last_inter_p)) + "\n")
            f.write("inter_n : " + str(sum(last_inter_n)/len(last_inter_n)) + "\n")
        
        # write the agreement ratio score
        ag_file = output_path + "last_ag_ratio.txt"
        with open(ag_file, "w+") as f:
            f.write(str(sum(last_ag_ratio)/len(last_ag_ratio)))

    except Exception as e:
        print(str(e))
        trace_str = traceback.format_exc()
        print(trace_str)
        for f in os.listdir(output_path):
            os.remove(os.path.join(output_path, f))
        with open(output_path + "/error_log.txt", "w+") as f:
            f.write(trace_str)
    
