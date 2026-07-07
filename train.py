import gc
import logging
import os
import random
import warnings

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from prettytable import PrettyTable
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from transformers import logging as transformers_logging

import util.dataloader4KGNN as dataloader4KGNN
import util.load_gat_db as load_gat
from model import MKG
from util.losses import MMLoss_Binary
from util.smiles_tokenizer import smiles_to_tensor

warnings.filterwarnings("ignore", category=FutureWarning)
transformers_logging.set_verbosity_error()

log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, mode="higher", patience=10, filename="checkpoint.pth"):
        self.mode = mode
        self.patience = patience
        self.filename = filename
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif (self.mode == "higher" and score <= self.best_score) or (
            self.mode == "lower" and score >= self.best_score
        ):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model, map_device):
        try:
            state_dict = torch.load(self.filename, map_location=map_device, weights_only=True)
        except TypeError:
            state_dict = torch.load(self.filename, map_location=map_device)
        model.load_state_dict(state_dict)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_cv_splits(cv_data, cv_mode=1, n_splits=5, seed=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if cv_mode == 1:
        for train_idx, val_idx in kf.split(cv_data):
            yield cv_data[train_idx], cv_data[val_idx]
    elif cv_mode == 2:
        unique_cells = np.unique(cv_data[:, 2])
        for train_idx, val_idx in kf.split(unique_cells):
            train_cells = unique_cells[train_idx]
            val_cells = unique_cells[val_idx]
            yield cv_data[np.isin(cv_data[:, 2], train_cells)], cv_data[np.isin(cv_data[:, 2], val_cells)]
    elif cv_mode == 3:
        pair_strings = np.array([f"{int(a)}_{int(b)}" for a, b in cv_data[:, 0:2]])
        unique_pairs = np.unique(pair_strings)
        for train_idx, val_idx in kf.split(unique_pairs):
            train_pairs = unique_pairs[train_idx]
            val_pairs = unique_pairs[val_idx]
            yield cv_data[np.isin(pair_strings, train_pairs)], cv_data[np.isin(pair_strings, val_pairs)]
    else:
        raise ValueError("cv_mode must be 1, 2, or 3.")


def logits_to_probs(logits):
    logits = logits.detach().cpu().float()
    if logits.ndim > 1:
        logits = logits[:, 1]
    return torch.sigmoid(logits).numpy()


def find_best_threshold(labels, logits, steps=101):
    labels = labels.detach().cpu().numpy()
    probs = logits_to_probs(logits)

    best_thresh = 0.5
    best_acc = -1.0
    for threshold in np.linspace(0.0, 1.0, int(steps)):
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = float(threshold)
    return best_thresh


def eval_classification(labels, logits, threshold=0.5):
    labels = labels.detach().cpu().numpy()
    probs = logits_to_probs(logits)
    predicted_label = (probs >= threshold).astype(int)

    try:
        auc_score = roc_auc_score(labels, probs)
    except ValueError:
        auc_score = float("nan")

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
        aupr = auc(recall_curve, precision_curve)
    except ValueError:
        aupr = float("nan")

    return {
        "precision": precision_score(labels, predicted_label, zero_division=0),
        "recall": recall_score(labels, predicted_label, zero_division=0),
        "accuracy": accuracy_score(labels, predicted_label),
        "auc": auc_score,
        "aupr": aupr,
        "f1": f1_score(labels, predicted_label, zero_division=0),
    }


def early_stop_score(metric_name, metrics, loss):
    if metric_name == "loss":
        return loss, "lower"
    if metric_name not in metrics:
        raise ValueError(f"Unknown early_stop_metric: {metric_name}")
    metric_value = metrics[metric_name]
    if np.isnan(metric_value):
        return loss, "lower"
    return metric_value, "higher"


def build_model(cfg, data_bundle):
    entitys = data_bundle["entitys"]
    relations = data_bundle["relations"]
    model_args = OmegaConf.merge(OmegaConf.create(), cfg.training, cfg.model)
    n_entitys = max(entitys) + 1
    n_relations = max(relations) + 1

    return MKG(
        model_args,
        n_entitys,
        n_relations,
        cfg.model.e_dim,
        cfg.model.r_dim,
        data_bundle["adj_entity"],
        data_bundle["adj_relation"],
        smiles_seq=data_bundle["sm_seq"],
        drug_emb_path=cfg.paths.drug_emb,
        cell_emb_path=cfg.paths.cell_emb,
        chemberta_dir=cfg.paths.chemberta_dir,
        smiles_path=cfg.paths.smiles_csv,
        cell_gene_path=cfg.paths.cell_gene_path,
    ).to(device)


def run_single_fold(fold_idx, cfg, data_bundle, train_set, val_set, test_set):
    log.info(f"\n======== Starting Fold {fold_idx} ========")
    log.info(f"Train size: {len(train_set)} | Val size: {len(val_set)} | Test size: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.training.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.training.batch_size * 2, shuffle=False)

    net = build_model(cfg, data_bundle)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.l2_weight)
    loss_fcn = nn.BCEWithLogitsLoss()
    mm_loss_fcn = MMLoss_Binary(tau=float(cfg.training.get("contrastive_tau", 0.07))).to(device)
    cl_alpha = float(cfg.training.get("cl_alpha", 0.005))
    aux_loss_weight = float(cfg.training.get("aux_loss_weight", 0.5))
    grad_clip_norm = float(cfg.training.get("grad_clip_norm", 10.0))
    threshold_steps = int(cfg.training.get("threshold_steps", 101))
    early_metric = cfg.training.get("early_stop_metric", "auc")
    scheduler_metric = cfg.training.get("scheduler_metric", "loss")
    scheduler_mode = "min" if scheduler_metric == "loss" else "max"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        scheduler_mode,
        patience=int(cfg.training.get("scheduler_patience", 5)),
        factor=float(cfg.training.get("scheduler_factor", 0.5)),
    )

    save_path = os.path.join(cfg.paths.save_dir, f"model_fold{fold_idx}.pkl")
    os.makedirs(cfg.paths.save_dir, exist_ok=True)
    stopper = EarlyStopping(mode="higher", patience=cfg.training.patience, filename=save_path)

    best_thresh = 0.5
    for epoch in range(cfg.training.n_epochs):
        net.train()
        total_train_loss = 0.0
        train_logits_list, train_labels_list = [], []

        for u1, u2, c, r, _ in train_loader:
            u1, u2, c, r = u1.to(device), u2.to(device), c.to(device), r.float().to(device)

            logits, f_feat, k_feat, logit_f, logit_k = net(u1, u2, c)
            task_loss = loss_fcn(logits, r)
            loss_aux_f = loss_fcn(logit_f, r)
            loss_aux_k = loss_fcn(logit_k, r)
            cl_loss = mm_loss_fcn(f_feat, k_feat, logit_f, logit_k, r)

            loss = task_loss + aux_loss_weight * (loss_aux_f + loss_aux_k) + cl_alpha * cl_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            total_train_loss += loss.item()
            train_logits_list.append(logits.detach().cpu())
            train_labels_list.append(r.detach().cpu())

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_metrics = eval_classification(torch.cat(train_labels_list), torch.cat(train_logits_list))

        net.eval()
        total_val_loss = 0.0
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for u1, u2, c, r, _ in val_loader:
                u1, u2, c, r = u1.to(device), u2.to(device), c.to(device), r.float().to(device)
                logits = net(u1, u2, c)
                total_val_loss += loss_fcn(logits, r).item()
                val_logits_list.append(logits.cpu())
                val_labels_list.append(r.cpu())

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        val_labels = torch.cat(val_labels_list)
        val_logits = torch.cat(val_logits_list)
        best_thresh = find_best_threshold(val_labels, val_logits, steps=threshold_steps)
        val_metrics = eval_classification(val_labels, val_logits, threshold=best_thresh)

        if scheduler_metric == "loss":
            scheduler_value = avg_val_loss
        else:
            scheduler_value = val_metrics.get(scheduler_metric, np.nan)
            if np.isnan(scheduler_value):
                scheduler_value = avg_val_loss
        scheduler.step(scheduler_value)

        if epoch % 5 == 0:
            table = PrettyTable(["Epoch", "Phase", "Loss", "ACC", "Pre", "Rec", "F1", "AUC", "AUPR"])
            table.float_format = ".4"
            table.add_row([
                epoch,
                "Train",
                avg_train_loss,
                train_metrics["accuracy"],
                train_metrics["precision"],
                train_metrics["recall"],
                train_metrics["f1"],
                train_metrics["auc"],
                train_metrics["aupr"],
            ])
            table.add_row([
                epoch,
                "Valid",
                avg_val_loss,
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                val_metrics["auc"],
                val_metrics["aupr"],
            ])
            log.info(f"\n{table}")
        else:
            log.info(f"[Fold {fold_idx} | Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val AUC: {val_metrics['auc']:.4f}")

        score, mode = early_stop_score(early_metric, val_metrics, avg_val_loss)
        stopper.mode = mode
        if stopper.step(score, net):
            log.info(f"Early stopping triggered at epoch {epoch}. Best {early_metric}: {stopper.best_score:.4f}")
            break

    log.info(f"Testing Fold {fold_idx} on hold-out test set using the best checkpoint...")

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    net = build_model(cfg, data_bundle)
    stopper.load_checkpoint(net, device)
    net.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for u1, u2, c, r, _ in test_loader:
            u1, u2, c = u1.to(device), u2.to(device), c.to(device)
            logits = net(u1, u2, c)
            all_logits.append(logits.cpu())
            all_labels.append(r.cpu())

    final_metrics = eval_classification(torch.cat(all_labels), torch.cat(all_logits), threshold=best_thresh)
    log.info(f"Fold {fold_idx} final hold-out results: {final_metrics}")

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_metrics


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_app(cfg: DictConfig):
    set_seed(cfg.training.seed)
    log.info(f"Output directory: {os.getcwd()}")
    log.info("Loading data...")

    _, _, _, triples = load_gat.readRecData(cfg.paths.rec_path)
    entitys, relations, kg_triples = load_gat.readKGData(cfg.paths.kg_path)
    n_entitys = max(entitys) + 1

    kg_indexes = dataloader4KGNN.getKgIndexsFromKgTriples(kg_triples)
    adj_entity, adj_relation = dataloader4KGNN.construct_adj(cfg.model.n_neighbors, kg_indexes, n_entitys)
    sm_seq = smiles_to_tensor(cfg.paths.smiles_csv, n_entitys)

    triples_np = np.array(triples)
    np.random.shuffle(triples_np)

    holdout_test_size = float(cfg.training.get("holdout_test_size", 0.1))
    cv_mode = int(cfg.training.get("cv_mode", 1))
    cv_data, test_data = train_test_split(triples_np, test_size=holdout_test_size, random_state=cfg.training.seed)
    test_set = test_data.tolist()

    data_bundle = {
        "entitys": entitys,
        "relations": relations,
        "adj_entity": adj_entity,
        "adj_relation": adj_relation,
        "sm_seq": sm_seq,
    }

    log.info(f"CV mode: {cv_mode}")
    log.info(f"Total: {len(triples_np)} | CV: {len(cv_data)} | Hold-out test: {len(test_set)}")

    metrics_all = {"precision": [], "recall": [], "accuracy": [], "auc": [], "aupr": [], "f1": []}
    for fold_idx, (train_data, val_data) in enumerate(
        generate_cv_splits(cv_data, cv_mode=cv_mode, n_splits=cfg.training.n_folds, seed=cfg.training.seed),
        start=1,
    ):
        fold_metrics = run_single_fold(fold_idx, cfg, data_bundle, train_data.tolist(), val_data.tolist(), test_set)
        for key, value in fold_metrics.items():
            metrics_all[key].append(value)
        gc.collect()

    log.info("=" * 80)
    log.info("Final cross-fold mean/std on hold-out test set")
    log.info(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}")
    log.info("-" * 40)
    for key, values in metrics_all.items():
        log.info(f"{key:<15} | {np.nanmean(values):.4f}      | {np.nanstd(values):.4f}")
    log.info("=" * 80)


if __name__ == "__main__":
    train_app()
