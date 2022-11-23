from typing import List
import torch
import numpy as np
from numpy.typing import ArrayLike
from torchmetrics.functional import retrieval_normalized_dcg


def num_relevant_at_k(actual, predicted, k):
    predicted = predicted[:k]

    hits = 0
    for item in predicted:
        if item in actual:
            hits += 1

    return hits


def precision_at_k(actual, predicted, k):
    return num_relevant_at_k(actual, predicted, k) / k


def ap_at_k(actual, predicted, k):
    """
    Average precision at K
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1.0
            score += hits / (i + 1.0)

    return score / min(len(actual), k)


def cumulative_gain(predictions: ArrayLike):
    array = np.asarray(predictions)
    return array.sum()


def discounted_cumulative_gain(predictions: ArrayLike):
    predictions = np.asarray(predictions)
    k = len(predictions)

    logs = np.log2(np.arange(k) + 2)  # THe logs are (i+1) starting with i=1
    dcg = (predictions / logs).sum()
    return dcg


def idcg(predictions: ArrayLike):
    """
    Ideal discounted cumulative gain
    """
    ordered = np.array(predictions)
    ordered.sort()
    # Descending
    ordered = ordered[::-1]
    return discounted_cumulative_gain(ordered)


def ndcg(predictions: ArrayLike):
    ideal = idcg(predictions)
    return discounted_cumulative_gain(predictions) / ideal


def ndcg_torch(predictions: List):
    """
    Alternative NDCG implementation in pytorch
    """
    target = predictions.copy()
    target.sort(reverse=True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    predictions = torch.tensor(predictions).to(device)
    target = torch.tensor(target).to(device)

    return retrieval_normalized_dcg(predictions, target)


if __name__ == "__main__":

    # MAP@K

    actual = np.array([1, 2, 3])
    predicted = np.array([2, 1, 5])

    assert 1 == precision_at_k(actual, predicted, k=1)
    assert 1 == precision_at_k(actual, predicted, k=2)
    assert 2 / 3 == precision_at_k(actual, predicted, k=3)

    assert 1 == ap_at_k(actual, predicted, k=1)
    assert 1 == ap_at_k(actual, predicted, k=2)
    assert 2 / 3 == ap_at_k(actual, predicted, k=3)

    # NDCG
    predictions = [0.1, 0.13, 0.35, 0.75]

    ndcg_np = ndcg(predictions)
    ndcg_pytorch = ndcg_torch(predictions)

    assert abs(ndcg_pytorch - ndcg_np) < 0.00001
