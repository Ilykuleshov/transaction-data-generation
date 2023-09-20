from typing import Literal
import torch

from sklearn.metrics import roc_auc_score, f1_score, r2_score


def roc_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return float(
        roc_auc_score(labels.detach().cpu().numpy(), probs.detach().cpu().numpy())
    )


def f1(
    preds: torch.Tensor,
    labels: torch.Tensor,
    average: Literal["micro", "macro", "samples", "weighted", "binary"] = "binary",
) -> float:
    return float(
        f1_score(
            labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average=average
        )
    )


def r2(preds: torch.Tensor, values: torch.Tensor) -> float:
    return float(r2_score(values.detach().cpu().numpy(), preds.detach().cpu().numpy()))
