
def merge(current_clustering, affinities):
    # Get pair of clusters with closest nearest neighbor, i.e. highest similarity
    min_index = affinities.argmin().item()
    cluster_a_index, cluster_b_index = (min_index // affinities.size(0),
                                        min_index % affinities.size(0))
    
    new_cluster_sample_indices = current_clustering[cluster_a_index] +\
        current_clustering[cluster_b_index]
    current_clustering = [current_clustering[cluster_index]
                          for cluster_index in range(len(current_clustering))
                          if cluster_index not in [cluster_a_index, cluster_b_index]]
    current_clustering.append(new_cluster_sample_indices)
    return current_clustering, cluster_a_index, cluster_b_index
