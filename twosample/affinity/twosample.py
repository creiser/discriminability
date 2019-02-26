import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import AlexNet, resnet18
from collections import Counter
from .. import utils
from ..datasets import ImbalancedMultiSampleDataset, ClusterIndexDataset
from timeit import default_timer as timer
import itertools
import numpy as np
from multiprocessing import Pool
import functools
from torch.utils.data.dataloader import default_collate

def twosample_affinity(current_clustering, dataset, hp):
    device = hp['device']
    hp = hp['affinity']

    # From all clusters a random subset will be taken to enforce that all clusters
    # have the same size.
    sample_indices_train = []
    sample_indices_test = []
    cluster_majority_categories, cluster_categories = [], []
    for sample_indices_i in current_clustering:
        # Train-test split
        sample_indices_i = utils.random_permutation(sample_indices_i) # IMPORTANT
        num_train_samples = int(len(sample_indices_i) * hp['train_ratio'])
        sample_indices_train.append(sample_indices_i[:num_train_samples])
        sample_indices_test.append(sample_indices_i[num_train_samples:])

    train_dataset = ImbalancedMultiSampleDataset(dataset, sample_indices_train)
    test_dataset = ImbalancedMultiSampleDataset(dataset, sample_indices_test)
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp['batch_size'],
                                               shuffle=True, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512,
                                              shuffle=False, pin_memory=True, num_workers=4)
        
    # Model
    #model = AlexNet(num_classes=len(current_clustering) ** 2)
    model = resnet18(num_classes=len(current_clustering) ** 2)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())

    # Train
    for epoch in range(1, hp['epochs'] + 1):
        model.train()
        total_loss = 0
        for data, target, weights, _ in train_loader:
            data, target, weights = data.to(device), target.to(device).view(target.size(0), -1),\
                                    weights.to(device).view(weights.size(0), -1)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy(F.sigmoid(output), target, reduce=False)
            loss *= weights # cost-senstive learning for class balance
            loss[target == -1] = 0 # ignore
            # Simply averaging is incorrect since we have to handle ignored targets properly
            columnwise_loss = loss.sum(dim=0)
            columnwise_denom = (target != -1).sum(dim=0).float()
            columnwise_denom[columnwise_denom == 0] = 1 # avoids divison by zero
            loss = columnwise_loss / columnwise_denom
            # Finally take the mean over columnwise losses
            loss = loss.mean()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        
        # Test
        model.eval()
        true_positives = torch.zeros([len(current_clustering), len(current_clustering)]).to(device)
        true_negatives = torch.zeros([len(current_clustering), len(current_clustering)]).to(device)
        with torch.no_grad():
            for data, target, _, _ in test_loader:
                data, target = data.to(device), target.to(device)
                output = F.sigmoid(model(data)).view(data.size(0),
                                                     len(current_clustering),
                                                     len(current_clustering))
                predictions = (output > 0.5).float()
                positive_target = target.clone()
                positive_target[target == 0] = -1
                negative_target = target.clone()
                negative_target[target == 1] = -1
                true_positives += (predictions == positive_target).float().sum(dim=0)
                true_negatives += (predictions == negative_target).float().sum(dim=0)

        # Calculate arithmetic means from true positives and true negatives
        arithmetic_means = torch.ones(len(current_clustering), len(current_clustering)).to(device)
        for cluster_i in range(len(current_clustering)):
            for cluster_j in range(len(current_clustering)):
                num_positive_test_samples = len(current_clustering[cluster_i]) -\
                                            int(len(current_clustering[cluster_i]) * hp['train_ratio'])
                num_negative_test_samples = len(current_clustering[cluster_j]) -\
                                            int(len(current_clustering[cluster_j]) * hp['train_ratio'])
                true_positive_rate = true_positives[cluster_i][cluster_j] / num_positive_test_samples
                true_negative_rate = true_negatives[cluster_i][cluster_j] / num_negative_test_samples
                arithmetic_means[cluster_i][cluster_j] = (true_positive_rate + true_negative_rate) / 2
        
        # Set to zero for mean calculation
        diagonal_indices = torch.arange(len(arithmetic_means), dtype=torch.long).to(device)
        arithmetic_means[diagonal_indices, diagonal_indices] = 0.
        print('Epoch: {}, Train loss: {:.2f}, mean arithmetic mean: {:.2f}'.format(epoch, total_loss, arithmetic_means.mean().item()))
        if False:
            for row in arithmetic_means.tolist():
                print(' '.join(['{:.2f}'.format(am) for am in row]))


    # Set diagonal to 1
    arithmetic_means[diagonal_indices, diagonal_indices] = 1.

    return arithmetic_means
    
def fast_twosample_affinity(current_clustering, dataset, hp):
    # Train-test split
    sample_indices_train, sample_indices_test = fast_twosample_train_test_split(current_clustering, hp)
    
    model = resnet18(num_classes=len(current_clustering))
    model = model.to(hp['device'])
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(1, hp['affinity']['epochs'] + 1):
        train_loss = fast_twosample_train(sample_indices_train, dataset, hp, model, optimizer)
        arithmetic_means = fast_twosample_test(sample_indices_test, dataset, hp, model)
        
        # Set diagonal to 0 for mean calculation
        diagonal_indices = torch.arange(len(arithmetic_means), dtype=torch.long).to(hp['device'])
        arithmetic_means[diagonal_indices, diagonal_indices] = 0.
        
        print('Epoch: {}, Train loss: {:.2f}, mean arithmetic mean: {:.2f}'.format(epoch, train_loss, arithmetic_means.mean().item()))
        
    # Set diagonal to 1
    arithmetic_means[diagonal_indices, diagonal_indices] = 1.
    return arithmetic_means
    

def fast_twosample_train_test_split(current_clustering, hp):
    sample_indices_train, sample_indices_test = [], []
    for sample_indices_i in current_clustering:
        sample_indices_i = utils.random_permutation(sample_indices_i) # IMPORTANT
        num_train_samples = int(len(sample_indices_i) * hp['affinity']['train_ratio'])
        sample_indices_train.append(sample_indices_i[:num_train_samples])
        sample_indices_test.append(sample_indices_i[num_train_samples:])
    return sample_indices_train, sample_indices_test
    

def fast_twosample_train(sample_indices_train, dataset, hp, model, optimizer):
    device = hp['device']
    cluster_sizes = [len(sample_indices_i) for sample_indices_i in sample_indices_train]
    cluster_sizes = torch.tensor(cluster_sizes, dtype=torch.long).to(device)
    num_clusters = len(sample_indices_train)

    train_dataset = ClusterIndexDataset(dataset, sample_indices_train) 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp['affinity']['batch_size'],
                                               shuffle=True, pin_memory=True, num_workers=4)
    # Train
    model.train()
    epoch_start_time = batch_end_time = timer()
    train_loss = 0
    load_times = []
    process_times = []
    for batch_number, (data, cluster_indices) in enumerate(train_loader):
        batch_start_time = timer()
        load_times.append(batch_start_time - batch_end_time)
        
        data, cluster_indices = data.to(device), cluster_indices.to(device)
        
        # Helper variables
        batch_size = data.size(0)
        batch_indices = list(range(batch_size))
        
        # Calculate weights for balanced loss from cluster sizes online.
        expanded_batch_cluster_sizes = cluster_sizes[cluster_indices].unsqueeze(1).expand(-1, num_clusters).float()
        expanded_cluster_sizes = cluster_sizes.unsqueeze(0).expand(batch_size, -1).float()
        prevalances = expanded_batch_cluster_sizes / (expanded_batch_cluster_sizes + expanded_cluster_sizes)
        weights = 1. / (prevalances * 2) # batch size x num clusters
        
        positive_target = torch.ones(batch_size, num_clusters).to(device)
        negative_target = torch.zeros(batch_size, num_clusters).to(device)
        
        # In total there are num clusters x num clusters tasks,
        # but a batch only contains samples that correspond to
        # a subset of these tasks. Each loss summand term corresponds
        # to a task.
        
        # Columns (negative)
        negative_frequencies = torch.zeros(num_clusters)
        for cluster_index in cluster_indices:
            negative_frequencies[cluster_index] += 1
        
        # Rows (positive)
        positive_frequencies = torch.tensor([negative_frequencies[cluster_index] for cluster_index in cluster_indices])
        positive_frequencies = positive_frequencies.unsqueeze(1).expand(-1, num_clusters)
        
        task_frequencies_per_loss_summand = (positive_frequencies + negative_frequencies.unsqueeze(0).expand(batch_size, -1)).view(-1)
        
        # Positive and negative loss summands have the same weight, 
        # therefore we just duplicate the positive loss summands' weights.
        task_frequencies_per_loss_summand = torch.cat([task_frequencies_per_loss_summand, task_frequencies_per_loss_summand]).to(device)
        
        # The task_indices set also contains the (0,0), (1,1), (2,2), .. diagonal tasks, which
        # are ignored. Therefore we have to subtract the number of these "degenerated" tasks
        num_unique_cluster_indices = len(set(cluster_indices.tolist()))
        num_tasks_in_batch = 2 * num_clusters * num_unique_cluster_indices - (num_unique_cluster_indices ** 2) - num_unique_cluster_indices
                     
        optimizer.zero_grad()
        output = model(data) # batch size x num clusters
        output_expanded = output[batch_indices, cluster_indices].unsqueeze(1).expand(-1, num_clusters) # batch size x num clusters
            
        # Row loss
        positive_loss = F.binary_cross_entropy(F.sigmoid(output_expanded - output), positive_target, reduce=False)
        positive_loss *= weights # cost-senstive learning to balance positive and negative class
        positive_loss[batch_indices, cluster_indices] = 0 # Ignore diagonal tasks.

        # Column loss (output and output_expanded are interchanged)
        negative_loss = F.binary_cross_entropy(F.sigmoid(output - output_expanded), negative_target, reduce=False)
        negative_loss *= weights # cost-senstive learning to balance positive and negative class
        negative_loss[batch_indices, cluster_indices] = 0 # Ignore diagonal tasks.
        
        # Flatten and concatenate positive and negative loss
        loss = torch.cat([positive_loss.view(-1), negative_loss.view(-1)]) # 2 * batch size * num clusters
        
        # Multiple loss summand terms belonging to the same task are contained in the batch:
        # Therefore we average the corresponding loss summands together.
        # It is more efficient to first divide and then sum together all the terms
        # instead of breaking them up into multiple mini sums.
        loss /= task_frequencies_per_loss_summand
        
        # Mean of the taskwise loss
        # Subtract 2 * batch_size because of the ignored targets.
        # We have to divide by the number of unique tasks in the batch, ignoring
        # the diagonal/degenerated tasks.
        loss = loss.sum() / num_tasks_in_batch
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if is_log_batch(batch_number, hp, train_loader):
            progress = (batch_number + 1) / len(train_loader)
            print('Batches processed: {}, ETA: {}, Loss: {:.6f}'.format(
                utils.pretty_accuracy(batch_number + 1, len(train_loader)),
                utils.get_eta(epoch_start_time, progress),
                loss.item()))
        
        batch_end_time = timer()
        process_times.append(batch_end_time - batch_start_time)
        #print('Load time:', sum(load_times) / len(load_times))
        #print('Process time:', sum(process_times) / len(process_times))

    return train_loss

def fast_twosample_test(sample_indices_test, dataset, hp, model):
    device = hp['device']
    num_clusters = len(sample_indices_test)
    
    test_dataset = ClusterIndexDataset(dataset, sample_indices_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                              shuffle=False, pin_memory=True, num_workers=4)

    # Test
    model.eval()
    
    # Finally we have to hold the num clusters x num clusters into memory, but only once
    # and not for every batch item. 
    true_positives = torch.zeros([num_clusters, num_clusters]).to(device)
    true_negatives = torch.zeros([num_clusters, num_clusters]).to(device)
    with torch.no_grad():
        epoch_start_time = timer()
        for batch_number, (data, cluster_indices) in enumerate(test_loader):
            data, cluster_indices = data.to(device), cluster_indices.to(device)
            
            # Helper variables
            batch_size = data.size(0)
            batch_indices = list(range(batch_size))
            
            output = model(data) # batch size x num clusters
            output_expanded = output[batch_indices, cluster_indices].unsqueeze(1).expand(-1, num_clusters) # batch size x num clusters
                
            true_positives_batch = (F.sigmoid(output_expanded - output) > 0.5).float()  # batch size x num clusters
            true_negatives_batch = (F.sigmoid(output - output_expanded) <= 0.5).float() # batch size x num clusters
            
            # TODO: Can we avoid the for loop?
            for i, cluster_index in enumerate(cluster_indices):
                # True positives must be added to the respective rows
                true_positives[cluster_index] += true_positives_batch[i]
                
                # True negatives must be added to the respective columns
                true_negatives[:, cluster_index] += true_negatives_batch[i]
            
            if is_log_batch(batch_number, hp, test_loader):
                progress = (batch_number + 1) / len(test_loader)
                print('Batches processed: {}, ETA: {}'.format(
                    utils.pretty_accuracy(batch_number + 1, len(test_loader)),
                    utils.get_eta(epoch_start_time, progress)))
        
        cluster_sizes = [len(sample_indices_i) for sample_indices_i in sample_indices_test]
        cluster_sizes = torch.tensor(cluster_sizes, dtype=torch.float).to(device)
        
        # Calculate arithmetic means from true positives and true negatives
        # Use in-place operations and broadcast semantics to save memory space.
        true_positive_rate = true_positives.div_(cluster_sizes.unsqueeze(1))
        true_negative_rate = true_negatives.div_(cluster_sizes.unsqueeze(0))
        arithmetic_means = (true_positive_rate.add_(true_negative_rate)).div_(2)
                    
    return arithmetic_means

def is_log_batch(batch_number, hp, loader):
    return (hp['affinity']['log_interval'] != -1) and\
        ((batch_number + 1) % hp['affinity']['log_interval'] == 0 or
         batch_number + 1 == len(loader))
