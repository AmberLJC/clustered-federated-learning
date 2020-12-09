#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:04:59 2020

@author: liujiachen
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from data_utils import split_noniid, CustomSubset


class ClientDataGenerater():
    def __init__(self, clients_num =10, num_group = 2, noniid_alpha = 1.0):
        # super().__init__( clients_num, num_group, noniid_alpha)
        self.clients_num = clients_num
        self.num_group = num_group 
        self.noniid_alpha = noniid_alpha
        
    
# fulltrainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    def load_emnist(self):
        data = datasets.EMNIST(root=".", split="byclass", download=True)
        idcs = np.random.permutation(len(data))
        self.train_idcs, self.test_idcs = idcs[:10000], idcs[10000:20000]
        self.train_labels = data.train_labels.numpy()
        self.client_idcs = split_noniid(self.train_idcs, self.train_labels, alpha=self.noniid_alpha, n_clients=self.clients_num)
        self.client_data = [CustomSubset(data, idcs) for idcs in self.client_idcs]
        self.test_data = CustomSubset(data, self.test_idcs, transforms.Compose([transforms.ToTensor()]))
        
        
    def rotate_noniid(self):
        # Rotate data for each group
        rotate_angle = 360 // self.num_group
        for i, client_datum in enumerate(self.client_data):
            group_idx = i // int(self.clients_num / self.num_group)
            client_datum.subset_transform = transforms.Compose([transforms.RandomRotation(
                    (rotate_angle * group_idx,rotate_angle * group_idx)),
                    transforms.ToTensor()])
        # TODO: need to shuffle clients' data
        return self.client_data, self.test_data
    
        
    def plot_distribution(self):
        self.mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
               'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
               'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
               'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'], dtype='<U1')
    
        plt.figure(figsize=(20,3))
        plt.hist([self.train_labels[idc]for idc in self.client_idcs], stacked=True, \
                 bins=np.arange(min(self.train_labels)-0.5, max(self.train_labels) + 1.5, 1),
        label=["Client {}".format(i) for i in range(self.clients_num)])
        plt.xticks(np.arange(62), self.mapp)
        plt.legend()
        plt.show()
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        