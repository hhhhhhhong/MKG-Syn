import numpy as np
import torch
import torch.nn.functional as F


class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature=0.5, use_cosine_similarity=True):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.mask_samples_from_same_repr = self._get_correlated_mask().bool()

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_similarity_fn
        return self._dot_similarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye(2 * self.batch_size, 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye(2 * self.batch_size, 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy(1 - (diag + l1 + l2))
        return mask.to(self.device)

    @staticmethod
    def _dot_similarity(x, y):
        return torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)

    def _cosine_similarity_fn(self, x, y):
        return self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))

    def forward(self, zis, zjs):
        if zis.shape[0] != self.batch_size or zjs.shape[0] != self.batch_size:
            raise ValueError(
                f"NTXentLoss expected batch size {self.batch_size}, "
                f"got {zis.shape[0]} and {zjs.shape[0]}."
            )

        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1) / self.temperature
        labels = torch.zeros(2 * self.batch_size, dtype=torch.long, device=self.device)
        loss = self.criterion(logits, labels)
        return loss / (2 * self.batch_size)


class MMLoss_Binary(torch.nn.Module):
    def __init__(self, tau=0.07, consistency_weight=0.1):
        super().__init__()
        self.tau = tau
        self.consistency_weight = consistency_weight

    def forward(self, feature_repr, knowledge_repr, feature_logits=None, knowledge_logits=None, labels=None):
        feature_repr = F.normalize(feature_repr, dim=-1)
        knowledge_repr = F.normalize(knowledge_repr, dim=-1)

        logits = torch.matmul(feature_repr, knowledge_repr.t()) / self.tau
        targets = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))

        if feature_logits is not None and knowledge_logits is not None:
            feature_prob = torch.sigmoid(feature_logits)
            knowledge_prob = torch.sigmoid(knowledge_logits)
            loss = loss + self.consistency_weight * F.mse_loss(feature_prob, knowledge_prob)

        return loss
