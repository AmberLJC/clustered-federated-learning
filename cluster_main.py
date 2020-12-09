#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:32:02 2020

@author: liujiachen
"""
from noniid_generator import ClientDataGenerater
import torch
from models import ConvNet
from helper import ExperimentLogger, display_train_stats
from fl_devices import Server, Client
import argparse
import matplotlib.pyplot as plt
import numpy as np

def visualize_client_data(clients):
    mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
               'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
               'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
               'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'], dtype='<U1')
    
    for client in [clients[0], clients[-1]]:
        x, y = iter(client.train_loader).next()
    
        print("Client {}:".format(client.id))
        plt.figure(figsize=(15,1))
        for i in range(10):
            plt.subplot(1,10,i+1)
            plt.imshow(x[i,0].numpy().T, cmap="Greys")
            plt.title("Label: {}".format(mapp[y[i].item()]))
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Simulate FL for noniid clients ')

    parser.add_argument('--num_client', type=int, default=10, 
                        help='number of clients') 
    parser.add_argument('--num_group', type=int, default=2, 
                        help='number of clients groups')
    parser.add_argument('--rounds', type=int, default=50, 
                        help='training rounds')  
    parser.add_argument('--local_epoch', type=int, default=1, 
                        help='local training epochs')  
    parser.add_argument('--noniid_alpha', type=float, default=1.0, 
                        help='DIRICHLET_ALPHA')     
    parser.add_argument('--frac', type=float, default=0.5, 
                        help='fraction of participants ')
    parser.add_argument('--eps1', type=float, default=0.4, 
                        help='fraction of participants ')
    parser.add_argument('--eps2', type=float, default=1.6, 
                        help='fraction of participants ')     
    
    args = parser.parse_args()
    print(args)
    # initialize data, clients, server, logger
    dataset = ClientDataGenerater(args.num_client, args.num_group, args.noniid_alpha)
    dataset.load_emnist()
    print('Finish loading data')

    noniid_train, noniid_test=  dataset.rotate_noniid()
    clients = [Client(ConvNet, lambda x : torch.optim.SGD(x, lr=0.1, momentum=0.9), dat, idnum=i) 
           for i, dat in enumerate(noniid_train)]
    server = Server(ConvNet, noniid_test)
    cfl_stats = ExperimentLogger()
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
    cfl_stats.log({'args':args})
    print('Finish setting up')
    
    local_epoch = args.local_epoch
    round_interval = 0
    for c_round in range(1, args.rounds+1):
        if c_round == 1:
            for client in clients:
                client.synchronize_with_server(server)
        participating_clients = server.select_clients(clients,frac = args.frac)
        
        # clients train locally
        for client in participating_clients:
            # average loss for a client
            train_stats = client.compute_weight_update(epochs=local_epoch)
            client.reset()
        similarities = server.compute_pairwise_similarities(clients)
        cluster_indices_new = []
        
        round_interval += 1
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            print(max_norm, mean_norm)
            # update avg gradient norm smaller than eps1: start converge
            # max norm of clients' gradient larger than eps2: need to split cluster
            if mean_norm<args.eps1 and max_norm>args.eps2 and len(idc)>2 :  # and round_interval > 20
                
                # server.cache_model(idc, clients[idc[0]].W, acc_clients)
                c1, c2 = server.cluster_clients(similarities[idc][:,idc]) 
                cluster_indices_new += [c1, c2]
                round_interval = 0
                cfl_stats.log({"split" : c_round})
    
            else:
                cluster_indices_new += [idc]
            
        cluster_indices = cluster_indices_new
        
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)
        acc_clients = [client.evaluate() for client in clients]
        log_dict = {"acc_clients" : acc_clients, "mean_norm" : mean_norm, "max_norm" : max_norm,
                      "rounds" : c_round, "clusters" : cluster_indices}
        cfl_stats.log(log_dict)
        print(log_dict)
        
    display_train_stats(cfl_stats, args.eps1, args.eps2, args.rounds)

    print("Clustering result: "  ,cluster_indices)
    for idc in cluster_indices:    
        server.cache_model(idc, clients[idc[0]].W, acc_clients)
        

















if __name__ == '__main__':
    main()