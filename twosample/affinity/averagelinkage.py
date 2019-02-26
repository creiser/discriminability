from .. import utils
import torch

def average_linkage_affinity(current_clustering, embeddings, hp):
    average_linkage = torch.zeros(len(current_clustering), len(current_clustering))
    for i, cluster_i in enumerate(current_clustering):
        for j, cluster_j in enumerate(current_clustering):
            if i <= j:
                continue
            average_linkage[i][j] = utils.pairwise_distances(embeddings[cluster_i],
                                                             embeddings[cluster_j]).mean()
            average_linkage[j][i] = average_linkage[i][j]

    # Set diagonal to infinity
    diagonal_indices = torch.arange(len(average_linkage), dtype=torch.long).to(hp['device'])
    average_linkage[diagonal_indices, diagonal_indices] = float('inf')
        
    return average_linkage
