#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    摘自gaussian_grouping
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.

    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.

    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]

    # Compute KL divergence
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    loss = kl.sum(dim=-1).mean()

    # Normalize loss into [0, 1]
    num_classes = predictions.size(1)
    normalized_loss = loss / num_classes

    return lambda_val * normalized_loss


def loss_cosine_similarity_3d(features, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Compute the loss for a 3D point cloud of a single class using cosine similarity.

    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling.
    :param sample_size: Number of points to randomly sample for computing the loss.

    :return: Computed loss value.

    特征归一化：我们首先对特征向量进行归一化处理，以确保它们位于同一比例尺上，这对于计算余弦相似度很重要。
    余弦相似度计算：我们计算每个采样点与所有点之间的余弦相似度。这是通过点特征的点积来完成的。
    选择最相似的邻居：我们选择每个采样点的Top-k最相似的邻居。
    损失计算：损失是基于最相似邻居的余弦相似度来计算的。在这个场景中，我们希望相似度尽可能高（即损失为负的平均相似度），这反映了点云内部一致性的高度。
    这个函数适用于处理单类别的点云数据，用于评估点云的内部结构一致性。通过考虑点之间的相似性，它有助于捕捉点云的空间特征。
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]

    # Normalize the feature vectors
    norm_features = F.normalize(features, p=2, dim=1)
    norm_sample_features = F.normalize(sample_features, p=2, dim=1)

    # Compute cosine similarity
    similarity_matrix = torch.mm(norm_sample_features, norm_features.t())

    # Find top-k similar neighbors
    _, neighbor_indices_tensor = similarity_matrix.topk(k, largest=True)

    # Select the similarities of the neighbors
    neighbor_similarities = similarity_matrix.gather(1, neighbor_indices_tensor)

    # Compute loss as negative average of the top-k similarities
    loss = -neighbor_similarities.mean()

    return lambda_val * loss

def loss_cosine_similarity_3d2(features, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Compute the loss for a 3D point cloud of a single class using cosine similarity.

    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling.
    :param sample_size: Number of points to randomly sample for computing the loss.

    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]

    # Normalize the feature vectors
    norm_features = F.normalize(features, p=2, dim=1)
    norm_sample_features = F.normalize(sample_features, p=2, dim=1)

    # Check dimensions and perform matrix multiplication
    if norm_features.dim() == 2 and norm_sample_features.dim() == 2:
        # Use torch.mm for 2D tensors
        similarity_matrix = torch.mm(norm_sample_features, norm_features.t())
    elif norm_features.dim() == 3 and norm_sample_features.dim() == 3:
        # Use torch.bmm for 3D tensors
        similarity_matrix = torch.bmm(norm_sample_features.unsqueeze(1), norm_features.transpose(1, 2)).squeeze(1)
    else:
        raise ValueError("Features dimensions are not compatible for matrix multiplication")

    # Find top-k similar neighbors
    _, neighbor_indices_tensor = similarity_matrix.topk(k, largest=True)

    # Select the similarities of the neighbors
    neighbor_similarities = similarity_matrix.gather(1, neighbor_indices_tensor)

    # Compute loss as negative average of the top-k similarities
    loss = -neighbor_similarities.mean()

    return lambda_val * loss

def loss_cosine_similarity_3d3(features, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Compute the loss for a 3D point cloud of a single class using cosine similarity.

    :param features: Tensor of shape (N, D) or (B, N, D), where N is the number of points, D is the dimensionality of the feature, and B is the batch size.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling.
    :param sample_size: Number of points to randomly sample for computing the loss.

    :return: Computed loss value.
    """
    # Ensure features is a 3D tensor (B, N, D)
    if features.dim() == 2:
        features = features.unsqueeze(0)  # Add batch dimension

    B, N, _ = features.size()

    # Conditionally downsample if points exceed max_points
    if N > max_points:
        indices = torch.randperm(N)[:max_points]
        features = features[:, indices, :]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(N)[:sample_size]
    sample_features = features[:, indices, :]

    # Normalize the feature vectors
    norm_features = F.normalize(features, p=2, dim=2)
    norm_sample_features = F.normalize(sample_features, p=2, dim=2)

    # Compute cosine similarity
    similarity_matrix = torch.bmm(norm_sample_features, norm_features.transpose(1, 2))

    # Find top-k similar neighbors
    _, neighbor_indices_tensor = similarity_matrix.topk(k, largest=True)

    # Select the similarities of the neighbors
    neighbor_similarities = torch.gather(similarity_matrix, 2, neighbor_indices_tensor.unsqueeze(-1)).squeeze(-1)

    # Compute loss as negative average of the top-k similarities
    loss = -neighbor_similarities.mean(dim=2).mean(dim=1)

    return lambda_val * loss.mean()

def loss_cosine_similarity_3d4(features, k=5, lambda_val=0.5, max_points=200000, sample_size=800):
    """
    Compute the loss for a 3D point cloud of a single class using cosine similarity.

    :param features: Tensor of shape (N, D) or (B, N, D), where N is the number of points, D is the dimensionality of the feature, and B is the batch size.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling.
    :param sample_size: Number of points to randomly sample for computing the loss.

    :return: Computed loss value.
    """
    # Ensure features is a 3D tensor (B, N, D)
    if features.dim() == 2:
        features = features.unsqueeze(0)  # Add batch dimension

    B, N, _ = features.size()

    # Conditionally downsample if points exceed max_points
    if N > max_points:
        indices = torch.randperm(N)[:max_points]
        features = features[:, indices, :]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(N)[:sample_size]
    sample_features = features[:, indices, :]

    # Normalize the feature vectors
    norm_features = F.normalize(features, p=2, dim=2)
    norm_sample_features = F.normalize(sample_features, p=2, dim=2)

    # Compute cosine similarity
    similarity_matrix = torch.bmm(norm_sample_features, norm_features.transpose(1, 2))
    # print(similarity_matrix.size())

    # Find top-k similar neighbors
    _, neighbor_indices_tensor = similarity_matrix.topk(k, largest=True)
    # print(neighbor_indices_tensor.size())

    # Expand neighbor_indices_tensor to use in gather
    # neighbor_indices_tensor = neighbor_indices_tensor.unsqueeze(-1).expand(-1, -1, -1, similarity_matrix.size(-1))
    # print(neighbor_indices_tensor.size())

    # Select the similarities of the neighbors
    neighbor_similarities = torch.gather(similarity_matrix, 2, neighbor_indices_tensor)

    # Compute loss as negative average of the top-k similarities
    loss = neighbor_similarities.mean()

    return lambda_val * loss.mean()

