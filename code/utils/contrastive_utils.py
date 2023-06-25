"""
We do not keep the cross-epoch memories while the feature prototypes are extracted in an online fashion
More details can be checked at https://github.com/Shathe/SemiSeg-Contrastive
Thanks the authors for providing such a model to achieve the class-level separation.
"""

import torch
import numpy as np
import random
import torch.nn.functional as F
from torch import nn


class FeatureMemory:

    def __init__(self, elements_per_class=32, n_classes=2):
        self.elements_per_class = elements_per_class
        self.memory = [None] * n_classes
        self.n_classes = n_classes

    def add_features_from_sample_learned(self, model, features, class_labels):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_selectors)
            features: Nx32 feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   Nx1  corresponding labels to the [features]
            batch_size: batch size
        Returns:
        """
        features = features.detach()
        class_labels = class_labels.detach().cpu().numpy()

        elements_per_class = self.elements_per_class

        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c  # get mask for class c
            selector = model.__getattr__(
                'contrastive_class_selector_' + str(c))  # get the self attention module for class c
            features_c = features[mask_c, :]  # get features from class c
            if features_c.shape[0] > 0:
                if features_c.shape[0] > elements_per_class:
                    with torch.no_grad():
                        # get ranking scores: Nx32(features_c) -> Nx1(rank) -> Nx1(rank)
                        rank = selector(features_c)
                        rank = torch.sigmoid(rank)
                        # sort them
                        _, indices = torch.sort(rank[:, 0], dim=0)
                        indices = indices.cpu().numpy()
                        features_c = features_c.cpu().numpy()
                        # get features with highest rankings: Nx32 -> 32x32
                        features_c = features_c[indices, :]
                        new_features = features_c[:elements_per_class, :]
                else:
                    # if not enough 32, directly feed into memory_bank
                    new_features = features_c.cpu().numpy()

                self.memory[c] = new_features


def generate_pseudo_labels(features, img_shape, num_classes, memory):
    """
     Args:
         features: Mx32  feature vectors for the contrastive learning (after applying the projection and prediction head)
         img_shape: A tuple of (B,D,H,W) before reshape
         num_classes: number of classes in the dataset
         memory: memory bank [List]
     Returns:
         returns generated pseudo-label
     """
    total_distances = []

    N, d, h, w = img_shape
    features = features.detach()  # M, 32
    features_norm = F.normalize(features, dim=1)  # M, 32->M, 32

    for c in range(num_classes):
        memory_c = memory[c]  # N=32, 32

        # At least memory is not empty
        if memory_c is not None and features_norm.shape[0] > 1 and memory_c.shape[0] > 1:
            memory_c = torch.from_numpy(memory_c).cuda()
            # L2 normalize vectors
            memory_c = F.normalize(memory_c, dim=1)  # N=32, 32

            # compute similarity. All elements with all elements
            # (M, 32) x (N=32, 32) -> M x N
            similarities = torch.mm(features_norm, memory_c.transpose(1, 0))  # MxN
            distances = 1 - torch.clamp(similarities, min=0)  # values between [0, 1] where 0 means same vectors
            # M (elements), N (memory) -> M -> M, 1
            distances, _ = torch.min(distances, dim=1)
            total_distances.append(distances.unsqueeze(1))
        else:
            # M, 1
            distances = torch.ones(features_norm.shape[0]).cuda()
            total_distances.append(distances.unsqueeze(1))

    # total_distances: num_classes x (M, 1) -> M, C
    total_distances = torch.cat(total_distances, dim=1)
    # pseudo_prob: M, C -> M, C
    pseudo_prob = F.softmin(total_distances, dim=1)

    # M, C -> M -> N, d, h, w
    weighted_map, pseudo_mask = torch.max(pseudo_prob, dim=1)
    pseudo_mask = pseudo_mask.reshape(N, d, h, w)
    weighted_map = weighted_map.reshape(N, d, h, w)
    return pseudo_mask, weighted_map

