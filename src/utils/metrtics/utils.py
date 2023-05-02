import torch

from sklearn.metrics import roc_auc_score, f1_score, r2_score


def roc_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return roc_auc_score(
        labels.detach().cpu().numpy(), probs.detach().cpu().numpy()
    )

def f1(preds: torch.Tensor, labels: torch.Tensor, average='binary') -> float:
    return f1_score(
        labels.detach().cpu().numpy(),
        preds.detach().cpu().numpy(),
        average=average
    )

def r2(preds: torch.Tensor, values: torch.Tensor) -> float:
    return r2_score(
        values.detach().cpu().numpy(), preds.detach().cpu().numpy()
    )
