from __future__ import print_function
import torch
from torchvision import transforms
from collections import Counter, defaultdict
import random
import copy
import numpy as np
from PIL import Image
    
class ImbalancedMultiSampleDataset(torch.utils.data.Dataset):
    # Sample indices will be split into equally sized portions
    def __init__(self, dataset, current_clusters):
        self.dataset = dataset
        self.sample_indices = []
        for sample_indices_i in current_clusters:
            self.sample_indices += sample_indices_i
        self.sample_indices = torch.tensor(self.sample_indices, dtype=torch.long)
        
        self.cluster_indices = []
        for cluster_index_i, sample_indices_i in enumerate(current_clusters):
            self.cluster_indices += [cluster_index_i for _ in range(len(sample_indices_i))]
        self.cluster_indices = torch.tensor(self.cluster_indices, dtype=torch.long)
        
        # Calculate the weights for the cost-sensitive loss function.
        # All samples from one cluster have the same weight matrix
        self.weights_per_cluster = torch.ones(len(current_clusters), len(current_clusters), len(current_clusters))
        for i in range(len(current_clusters)):
            for j in range(len(current_clusters)):
                num_positive_samples = len(current_clusters[i])
                num_negative_samples = len(current_clusters[j])
                positive_prevalence = num_positive_samples / (num_positive_samples + num_negative_samples)
                self.weights_per_cluster[i, i, j] = 1. / (positive_prevalence * 2)
                self.weights_per_cluster[i, j, i] = 1. / (positive_prevalence * 2)
                # These two lines are just for debug reasons
                #self.weights_per_cluster[i, i, j] = positive_prevalence
                #self.weights_per_cluster[i, j, i] = positive_prevalence
            self.weights_per_cluster[i, i, i] = 1.
         
        # In the i-th row and j-th column is the classifier between cluster i and j
        # where cluster i is defined to be the positve class and cluster j the negative class
        # All samples from one cluster have the same label
        self.labels_per_cluster = torch.full((len(current_clusters), len(current_clusters), len(current_clusters)), -1)
        for i in range(len(current_clusters)):
            self.labels_per_cluster[i, i]    = 1 # i-th row:    i-th cluster positive
            self.labels_per_cluster[i, :, i] = 0 # i-th column: i-th cluster negative
            self.labels_per_cluster[i, i, i] = -1
        

    def __getitem__(self, index):
        return self.dataset[self.sample_indices[index]][0], self.labels_per_cluster[self.cluster_indices[index]], self.weights_per_cluster[self.cluster_indices[index]], self.cluster_indices[index]
    
    def __len__(self):
        return len(self.sample_indices)

# This dataset merely returns the cluster_index of a sample as a target.
# Weights and actual targets wil be computed on-the-fly.
class ClusterIndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, current_clusters):
        self.dataset = dataset
        self.sample_indices = []
        for sample_indices_i in current_clusters:
            self.sample_indices += sample_indices_i
        self.sample_indices = torch.tensor(self.sample_indices, dtype=torch.long)
        
        self.cluster_indices = []
        for cluster_index_i, sample_indices_i in enumerate(current_clusters):
            self.cluster_indices += [cluster_index_i for _ in range(len(sample_indices_i))]
        self.cluster_indices = torch.tensor(self.cluster_indices, dtype=torch.long)
    
    def __getitem__(self, index):
        return self.dataset[self.sample_indices[index]][0], self.cluster_indices[index]
    
    def __len__(self):
        return len(self.sample_indices)
