import torch
import torch.nn.functional as F


def contrastive_class_to_class_learned_memory(model, features, class_labels, num_classes, memory):
    """
    Args:
        model: segmentation model that contains the self-attention MLPs for selecting the features
        to take part in the contrastive learning optimization
        features: Nx32  feature vectors for the contrastive learning (after applying the projection and prediction head)
        class_labels: N corresponding class labels for every feature vector
        num_classes: number of classes in the dataset
        memory: memory bank [List]
    Returns:
        returns the contrastive loss between features vectors from [features] and from [memory] in a class-wise fashion.
    """

    loss = 0
    for c in range(num_classes):
        # get features of an specific class
        # mask_c: N x 1 -> Nc x 1; features_c: N x 32 -> Nc x 32
        mask_c = class_labels == c
        features_c = features[mask_c, :]
        memory_c = memory[c]  # 32, 32

        # get the self-attention MLPs both for memory features vectors (projected vectors)
        # and network feature vectors (predicted vectors)

        selector = model.__getattr__('contrastive_class_selector_' + str(c))
        selector_memory = model.__getattr__('contrastive_class_selector_memory' + str(c))

        # At least class c has one correctly-classified pixel and memory is not empty
        if memory_c is not None and features_c.shape[0] > 1 and memory_c.shape[0] > 1:
            memory_c = torch.from_numpy(memory_c).cuda()

            # L2 normalize vectors
            memory_c = F.normalize(memory_c, dim=1)  # M=32, 32
            features_c_norm = F.normalize(features_c, dim=1)  # N(Nc), 32

            # compute similarity. All elements with all elements
            similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0))  # MxN
            distances = 1 - similarities  # values between [0, 2] where 0 means same vectors
            # M (elements), N (memory)

            # now weight every sample: M, 32 -> M, 1; N, 32 -> N, 1
            learned_weights_features = selector(features_c.detach())  # detach for trainability
            learned_weights_features_memory = selector_memory(memory_c)

            # self-attention in the memory features-axis and on the learning contrastive features-axis
            learned_weights_features = torch.sigmoid(learned_weights_features)
            rescaled_weights = (learned_weights_features.shape[0] / learned_weights_features.sum(
                dim=0)) * learned_weights_features
            # rescaled_weights: M, 1 -> M, N; distances: M, N * M, N -> M, N
            rescaled_weights = rescaled_weights.repeat(1, distances.shape[1])
            distances = distances * rescaled_weights

            learned_weights_features_memory = torch.sigmoid(learned_weights_features_memory)
            learned_weights_features_memory = learned_weights_features_memory.permute(1, 0)
            # rescaled_weights_memory: N, 1 -> 1, N -> M, N; distances: M, N * M, N -> M, N
            rescaled_weights_memory = (learned_weights_features_memory.shape[0] / learned_weights_features_memory.sum(
                dim=0)) * learned_weights_features_memory
            rescaled_weights_memory = rescaled_weights_memory.repeat(distances.shape[0], 1)
            distances = distances * rescaled_weights_memory

            loss = loss + distances.mean()

    return loss / num_classes

