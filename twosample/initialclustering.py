
import os.path
from . import utils
from .embedding import get_embeddings
import numpy as np
import random
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter

def get_initial_clustering(hp):
    initial_clustering_directory = 'results/2-initialclustering/{}'.format(hp['dataset'])
    utils.create_directory_if_not_exists(initial_clustering_directory)
    initial_clustering_filename = initial_clustering_directory + '/{}x{}.npy'.format(
        hp['initial_clustering']['num_initial_clusters'],
        hp['initial_clustering']['initial_cluster_size'])
     
    if os.path.isfile(initial_clustering_filename):
        print('Loading initial clustering from {}'.format(initial_clustering_filename))
        current_clustering = np.load(initial_clustering_filename).tolist()
    else:
        embeddings = get_embeddings(hp)                       
        embeddings_unassigned = embeddings.clone()
        current_clustering = []
    
        hp = hp['initial_clustering']
        while len(current_clustering) < hp['num_initial_clusters']:
            distances = utils.pairwise_distances(embeddings_unassigned)
            nearest_neighbors = distances.topk(hp['initial_cluster_size'], largest=False)
            cluster_tightness = nearest_neighbors[0].mean(dim=1)
            tightest_cluster = cluster_tightness.argmin().item()
            new_cluster_indices = nearest_neighbors[1][tightest_cluster]
            current_clustering.append(new_cluster_indices.tolist())
            embeddings_unassigned[new_cluster_indices] = float('inf')
        
        # Write clusters to disk such that they are accessible for subsequent merging.
        print('Writing initial clustering to ' + initial_clustering_filename)
        np.save(initial_clustering_filename, current_clustering)
    return current_clustering

def get_artifical_initial_clustering(hp, dataset):
    categories = utils.get_categories(dataset, range(len(dataset)))
    current_clustering = []
    for category in set(categories):
        sample_indices = [i for i, cat in enumerate(categories) if cat == category]
        num_clusters_for_this_category = len(sample_indices) // hp['initial_clustering']['minimum_initial_cluster_size']
        # Evenly split samples across clusters
        for sample_indices_i in np.array_split(sample_indices, num_clusters_for_this_category):
            current_clustering.append(sample_indices_i.tolist())
            
    # Optionally add noise to the initial clustering.
    if hp['initial_clustering']['noise_ratio'] > 0:
        num_samples_to_flip = int(hp['initial_clustering']['noise_ratio'] * len(dataset))
        flipped_sample_indices = set(random.sample(range(len(dataset)), num_samples_to_flip))
        
        # Remove the flipped samples first.
        current_clustering = [[sample_index for sample_index in sample_indices_i
            if sample_index not in flipped_sample_indices] for sample_indices_i in current_clustering]
        
        # Add the flipped samples to a different cluster.
        for flipped_sample_index in flipped_sample_indices:
            # Make sure that the sample is moved to cluster with a different majority category,
            # so it is guranteed to be in an "incorrect" cluster.
            while True:
                target_cluster_index = random.randint(0, len(current_clustering) - 1)
                target_majority_category = utils.get_majority_category(dataset, current_clustering[target_cluster_index])
                if categories[flipped_sample_index] != target_majority_category:
                    break
            current_clustering[target_cluster_index].append(flipped_sample_index)  
    
    return current_clustering
    
    
