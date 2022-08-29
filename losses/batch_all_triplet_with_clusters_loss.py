import numpy as np
import torch
import torch.nn.functional as F

eps = 1e-8  # an arbitrary small value to be used for numerical stability tricks


def euclidean_distance_matrix(x):
    """Efficient computation of Euclidean distance matrix

    Args:
      x: Input tensor of shape (batch_size, embedding_dim)

    Returns:
      Distance matrix of shape (batch_size, batch_size)
    """
    # step 1 - compute the dot product

    # shape: (batch_size, batch_size)
    dot_product = torch.mm(x, x.t())

    # step 2 - extract the squared Euclidean norm from the diagonal

    # shape: (batch_size,)
    squared_norm = torch.diag(dot_product)

    # step 3 - compute squared Euclidean distances

    # shape: (batch_size, batch_size)
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)

    # get rid of negative distances due to numerical instabilities
    distance_matrix = F.relu(distance_matrix)

    # step 4 - compute the non-squared distances

    # handle numerical stability
    # derivative of the square root operation applied to 0 is infinite
    # we need to handle by setting any 0 to eps
    mask = (distance_matrix == 0.0).float()

    # use this mask to set indices with a value of 0 to eps
    distance_matrix += mask * eps

    # now it is safe to get the square root
    # distance_matrix = torch.sqrt(distance_matrix)

    # undo the trick for numerical stability
    distance_matrix *= (1.0 - mask)

    return distance_matrix


def get_quadruplet_mask(labels, clusters):
    """compute a mask for valid triplets
    Args:
      labels: Batch of integer labels. shape: (batch_size,)
    Returns:
      Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
      A triplet is valid if:
      `labels[i] == labels[j] and labels[i] != labels[k]`
      and `i`, `j`, `k` are different.
    """
    # step 1 - get a mask for distinct indices
    # shape: (batch_size, batch_size)
    indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
    indices_not_equal = torch.logical_not(indices_equal)
    # shape: (batch_size, batch_size, 1)
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    # shape: (1, batch_size, batch_size)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    # Shape: (batch_size, batch_size, batch_size)
    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # step 2 - get a mask for valid anchor-positive-negative triplets

    # shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    # shape: (batch_size, batch_size, 1)
    i_equal_j_label = labels_equal.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    i_equal_k_label_first_example = labels_equal.unsqueeze(1)
    i_equal_k_label_second_example = i_equal_k_label_first_example.detach().clone()
    # shape: (batch_size, batch_size, batch_size)

    # step 3 - gennerate arrays for clusters
    clusters_equal = clusters.unsqueeze(0) == clusters.unsqueeze(1)

    # shape: (batch_size, batch_size, 1)
    i_equal_j_clusters = clusters_equal.unsqueeze(2)

    i_not_equal_j_clusters = torch.logical_not(i_equal_j_clusters)

    # shape: (batch_size, batch_size, batch_size)

    # EXPERIMENTAL SITE ---------------------------------------------------------------

    for row_idx, _ in enumerate(i_equal_k_label_first_example):

        true_indices = i_equal_k_label_first_example[row_idx][0] == True
        indices = true_indices.nonzero()

        first_selection = []
        second_selection = []

        for idx, torch_idx in enumerate(indices):
            if idx % 2 == 0:
                first_selection.append(int(torch_idx))
            else:
                second_selection.append(int(torch_idx))

        for selected_id in first_selection:
            i_equal_k_label_first_example[row_idx][0][selected_id] = False

        for selected_id in second_selection:
            i_equal_k_label_second_example[row_idx][0][selected_id] = False

    # EXPERIMENTAL SITE ---------------------------------------------------------------

    valid_anch_pos_in_neg_indices = torch.logical_and(torch.logical_and(i_equal_j_label, i_equal_j_clusters),
                                                      torch.logical_not(i_equal_k_label_first_example))

    valid_anch_pos_out_neg_indices = torch.logical_and(torch.logical_and(i_equal_j_label, i_not_equal_j_clusters),
                                                       torch.logical_not(i_equal_k_label_second_example))

    # step 3 - combine two masks
    pos_in_cluster_mask = torch.logical_and(distinct_indices, valid_anch_pos_in_neg_indices)
    pos_out_cluster_mask = torch.logical_and(distinct_indices, valid_anch_pos_out_neg_indices)

    return pos_in_cluster_mask, pos_out_cluster_mask


class BatchAllTripletWithClustersLossSemiHard(torch.nn.Module):
    """Uses all valid triplets to compute Triplet loss
    Args:
      margin: Margin value in the Triplet Loss equation
    """

    def __init__(self, weights, configuration, margin=1.):
        super().__init__()
        self.margin = margin
        self.weights = weights
        self.configuration = configuration

    def forward(self, embeddings, labels, clusters):
        """computes loss value.
        Args:
          embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
          labels: Batch of integer labels associated with embeddings. shape: (batch_size,)
        Returns:
          Scalar loss value.
        """
        # step 1 - get distance matrix
        # shape: (batch_size, batch_size)
        distance_matrix = euclidean_distance_matrix(embeddings)

        # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix

        # shape: (batch_size, batch_size, 1)
        anchor_positive_inside_cluster_dists = distance_matrix.unsqueeze(2)

        # shape: (batch_size, batch_size, 1)
        anchor_positive_outside_cluster_dists = distance_matrix.unsqueeze(2)

        # shape: (batch_size, 1, batch_size)
        anchor_negative_dists = distance_matrix.unsqueeze(1)
        # get loss values for all possible n^3 triplets
        # shape: (batch_size, batch_size, batch_size)

        first_factor = anchor_positive_inside_cluster_dists - anchor_negative_dists + self.margin
        second_factor = anchor_positive_outside_cluster_dists - anchor_negative_dists + self.margin

        # step 3 - filter out invalid or easy triplets by setting their loss values to 0

        # shape: (batch_size, batch_size, batch_size)
        pos_in_mask, pos_out_mask = get_quadruplet_mask(labels, clusters)

        first_factor *= pos_in_mask
        second_factor *= pos_out_mask

        quadruplet_loss = F.relu(first_factor) + F.relu(second_factor)

        # if weights are enabled, multiply them with the loss
        if self.configuration['weights_enabled'] == "True":
            target_weights = []
            for t in labels:
                target_weights.append(self.weights[t.item()])
            w = torch.Tensor(np.array(target_weights))

            quadruplet_loss = quadruplet_loss * torch.reshape(w, [len(w), 1])
        else:
            # easy triplets have negative loss values
            quadruplet_loss = quadruplet_loss
        # step 4 - compute scalar loss value by averaging positive losses
        num_positive_losses = (quadruplet_loss > eps).float().sum()
        quadruplet_loss = quadruplet_loss.sum() / (num_positive_losses + eps)
        return quadruplet_loss
