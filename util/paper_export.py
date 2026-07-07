import os

import pandas as pd
import torch


def _to_flat_cpu(x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return x.detach().cpu().view(-1)


def save_test_predictions(
    path,
    drug1_ids,
    drug2_ids,
    cell_ids,
    labels,
    logits,
    folds=None,
):
    drug1_ids = _to_flat_cpu(torch.cat(drug1_ids) if isinstance(drug1_ids, list) else drug1_ids)
    drug2_ids = _to_flat_cpu(torch.cat(drug2_ids) if isinstance(drug2_ids, list) else drug2_ids)
    cell_ids = _to_flat_cpu(torch.cat(cell_ids) if isinstance(cell_ids, list) else cell_ids)
    labels = _to_flat_cpu(torch.cat(labels) if isinstance(labels, list) else labels)
    logits = _to_flat_cpu(torch.cat(logits) if isinstance(logits, list) else logits)

    data = {
        "drug1_id": drug1_ids.numpy().astype(int),
        "drug2_id": drug2_ids.numpy().astype(int),
        "cell_id": cell_ids.numpy().astype(int),
        "label": labels.numpy().astype(float),
        "logit": logits.numpy().astype(float),
        "predicted_prob": torch.sigmoid(logits).numpy().astype(float),
    }

    if folds is not None:
        folds = _to_flat_cpu(torch.cat(folds) if isinstance(folds, list) else folds)
        data["fold"] = folds.numpy().astype(int)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8-sig")
    return path
