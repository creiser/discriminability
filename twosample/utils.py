import os
import random
from collections import Counter
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
from timeit import default_timer as timer
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

# Returns the value and frequency of the most common element of a list
def get_most_common(l):
    return Counter(l).most_common(1)[0]

def random_subset(l, size):
    return [l[i] for i in random.sample(range(len(l)), size)]
    
def random_permutation(l):
    return random_subset(l, len(l))

def flatten_list(l):
    return [item for sublist in l for item in sublist]

# Returns True if all elements are equal, otherwise False   
def check_equal(l):
   return len(set(l)) <= 1
   
def get_categories(dataset, sample_indices):
    if type(dataset) is datasets.ImageFolder:
        return [dataset.imgs[sample_index][1] for sample_index in sample_indices]
    return [tensor_item(dataset[sample_index][1]) for sample_index in sample_indices]

# Sorts a dictionary by value
def sorted_dict(d):
    return [(k, d[k]) for k in sorted(d, key=d.get, reverse=True)]
    
def pretty_accuracy(num_correct, num_total):
    return '{}/{} ({:.1f}%)'.format(int(num_correct), num_total, 100. * num_correct / num_total)
    
def create_directory_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

    
def time_string():
    return datetime.datetime.now()


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

    
def tensor_item(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return x
    
def print_purity(dataset, current_clustering):
    count_majority = count_total = 0
    for sample_indices_i in current_clustering:
        cluster_i_categories = Counter(get_categories(dataset, sample_indices_i))
        count_majority += cluster_i_categories.most_common(1)[0][1]
        count_total += len(sample_indices_i)
    print('Purity:', pretty_accuracy(count_majority, count_total))
    
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def balanced_bce_ignore(output, target, weights):
    loss = F.binary_cross_entropy(output, target, reduce=False)
    loss *= weights # cost-senstive learning for class balance
    loss[target == -1] = 0 # ignore
    # Simply averaging is incorrect since we have to handle ignored targets properly
    columnwise_loss = loss.sum(dim=0)
    columnwise_denom = (target != -1).sum(dim=0).float()
    columnwise_denom[columnwise_denom == 0] = 1 # avoids divison by zero
    loss = columnwise_loss / columnwise_denom
    # Finally take the mean over columnwise losses
    return loss.mean()
    

def mixup_data(data, target, weights, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, pairs of weights and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size(0)
    permutation = torch.randperm(batch_size).to(device)

    mixed_data = lam * data + (1 - lam) * data[permutation, :]
    target_a, target_b = target, target[permutation]
    weights_a, weights_b = weights, weights[permutation]
    return mixed_data, target_a, target_b, weights_a, weights_b, lam


def mixup_criterion(criterion, output, target_a, target_b, weights_a, weights_b, lam):
    return lam * criterion(output, target_a, weights_a) +\
     (1 - lam) * criterion(output, target_b, weights_b)

    
def pretty_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    return '\n'.join(table)

def get_majority_category(dataset, sample_indices):
    return Counter(get_categories(dataset, sample_indices)).most_common(1)[0][0]
    
def pretty_duration(seconds):
    seconds = round(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    res = ''
    if h != 0:
        res += '{}h:'.format(int(h))
    if h != 0 or m != 0:
        res += '{0:02}m:'.format(int(m))
    if h == 0 and m == 0 and s < 10:
        res += '{0:01}s'.format(s)
    else:
        res += '{0:02}s'.format(s)
    return res

def get_eta(start_time, progress):
    elapsed_time = timer() - start_time
    estimated_total_time = elapsed_time / progress
    remaining_time = estimated_total_time - elapsed_time
    return pretty_duration(remaining_time)
    
def get_labels_true_and_labels_pred(dataset, current_clustering):
    sample_indices = flatten_list(current_clustering)
    labels_true = get_categories(dataset, sample_indices)
    labels_pred = []
    for cluster_index, sample_indices in enumerate(current_clustering):
        labels_pred += [cluster_index] * len(sample_indices)
    return labels_true, labels_pred
    
def nmi(dataset, current_clustering):
    labels_true, labels_pred = get_labels_true_and_labels_pred(dataset, current_clustering)
    return normalized_mutual_info_score(labels_true, labels_pred)

def ami(dataset, current_clustering):
    labels_true, labels_pred = get_labels_true_and_labels_pred(dataset, current_clustering)
    return adjusted_mutual_info_score(labels_true, labels_pred)

def remove_list_elements(l, indices):
    for index in sorted(indices, reverse=True):
        del l[index]

    
