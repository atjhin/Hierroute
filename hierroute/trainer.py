from __future__ import annotations

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import HierRouteNet

class Trainer:
    """
    Training and evaluation engine for HierRouteNet.

    Handles the full training loop (with early stopping), inference over a DataLoader,
    per-level hierarchical evaluation, and structural consistency checking.

    Attributes:
        learning_rate     (float):      Initial learning rate for Adam.
        max_epochs        (int):        Maximum number of training epochs.
        gradient_clip_val (float):      Max gradient norm for clipping; 0 disables clipping.
        device            (str):        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).
        model_dir         (str | None): Directory for saving checkpoints and metadata.
                                        A numeric suffix is appended if it already exists.
        print_every       (int):        Log progress every this many epochs (default 5).
        train_loss        (list[float]):Per-epoch average training loss.
        valid_loss        (list[float]):Per-epoch average validation loss.
        train_acc         (list[float]):Per-epoch training accuracy (leaf level).
        valid_acc         (list[float]):Per-epoch validation accuracy (leaf level).
        train_f1          (list[float]):Per-epoch training macro-F1 (leaf level).
        valid_f1          (list[float]):Per-epoch validation macro-F1 (leaf level).
    """

    def __init__(self, learning_rate: float, max_epochs: int, gradient_clip_val: float = 1,
                 device: str = "cpu", print_every: int = 5, model_dir: str = None) -> None:
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.device = device
        self.model_dir = model_dir
        self.train_loss = []  
        self.valid_loss = []  
        self.train_acc = []
        self.valid_acc = []
        self.train_f1 = []
        self.valid_f1 = []
        self.print_every = print_every

    @staticmethod
    def clip_gradients(model: "HierRouteNet", max_norm: float) -> None:
        """
        Clip the gradient norm of all model parameters in-place.

        Args:
            model    (HierRouteNet): The model whose gradients to clip.
            max_norm (float):        Maximum allowed L2 norm. Pass 0 or None to skip clipping.
        """
        if max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    @staticmethod
    def _get_level_predictions(
        model: HierRouteNet,
        pred_paths: list[list[int]],
        true_ids: torch.Tensor,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Aligns predictions and true labels at each hierarchy depth.
        Returns {depth: (preds_array, trues_array)} using the same
        logic as evaluate(), for saving and downstream visualisation.
        """
        true_list = true_ids.tolist()
        max_depth = max(node.depth for node in model.hierarchy.nodes.values())
        level_arrays = {}
        for depth in range(1, max_depth + 1):
            preds_at_depth, trues_at_depth = [], []
            for i, true_id in enumerate(true_list):
                if model.hierarchy.nodes[true_id].depth < depth:
                    continue
                true_path = model.hierarchy.get_path_to_root(true_id)
                pred_path = pred_paths[i]
                trues_at_depth.append(true_path[depth])
                preds_at_depth.append(pred_path[depth] if depth < len(pred_path) else pred_path[-1])
            level_arrays[depth] = (np.array(preds_at_depth), np.array(trues_at_depth))
        return level_arrays

    @staticmethod
    def compute_metrics(
        leaf_probs: torch.Tensor,
        label_node: torch.Tensor,
        leaf_index: torch.Tensor,
    ) -> tuple[float, float]:
        """
        Converts soft predictions and targets to hard class predictions
        over leaf nodes only, then computes accuracy and macro F1.

        Args:
            leaf_probs  : (B, N) predicted probabilities
            label_node  : (B, N) soft ground truth labels
            leaf_index  : (N,)   binary mask — 1 for leaf nodes

        Returns:
            accuracy (float), f1 (float)
        """
        leaf_mask = leaf_index.bool()

        leaf_probs_only  = leaf_probs[:, leaf_mask]
        label_node_only  = label_node[:, leaf_mask]

        pred_classes   = leaf_probs_only.argmax(dim=1).cpu()
        target_classes = label_node_only.argmax(dim=1).cpu()

        accuracy = (pred_classes == target_classes).float().mean().item()
        f1 = f1_score(target_classes.numpy(), pred_classes.numpy(), average="macro", zero_division=0)

        return accuracy, f1


    def fit(
        self,
        model: HierRouteNet,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: bool = False,
        patience: int = 15,
        delta: float = 0.0005,
    ) -> None:
        """
        Train the model with early stopping on validation accuracy.

        Args:
            model        : HierRouteNet
            train_loader : DataLoader for training set
            valid_loader : DataLoader for validation set
            optimizer    : Optional pre-built optimizer; Adam is used if None
            scheduler    : If True, uses CosineAnnealingLR with T_max=self.max_epochs (default: False)
            patience     : Early stopping patience in epochs (default: 15)
            delta        : Minimum improvement in valid accuracy to count as a new best (default: 0.0005)

        Saves best_model.pt and training_info.json to self.model_dir if set. If the directory
        already exists a numeric suffix is appended (_1, _2, …) to avoid overwriting.
        """
        print(f"Training at device: {self.device}")
        model.to(self.device)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        if scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        else:
            scheduler = None

        # --- Output directory with collision safety ---
        actual_dir = None
        if self.model_dir is not None:
            actual_dir = self.model_dir
            suffix = 1
            while os.path.exists(actual_dir):
                actual_dir = f"{self.model_dir}_{suffix}"
                suffix += 1
            os.makedirs(actual_dir)
            if actual_dir != self.model_dir:
                print(f"Directory '{self.model_dir}' already exists — saving to '{actual_dir}' instead.")
                self.model_dir = actual_dir
            best_model_path = os.path.join(actual_dir, "best_model.pt")

        best_val_acc      = -float("inf")
        best_epoch        = -1
        epochs_no_improve = 0

        print("beginning training")
        leaf_mask = model.leaf_index.bool()
        for epoch in range(self.max_epochs):
            model.train()
            train_loss = 0
            train_preds, train_targets = [], []
            
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                image, label_node, path, targets, masks = batch.values()
                image, label_node = image.to(self.device), label_node.to(self.device)
                leaf_tensor, node_logits = model(image)
                
                loss = model.loss_fn(leaf_tensor, label_node)
                loss.backward()
                self.clip_gradients(model, self.gradient_clip_val)
                optimizer.step()
                train_loss += loss.detach().cpu().item()

                train_preds.append(leaf_tensor[:, leaf_mask].argmax(dim=1).detach().cpu())
                train_targets.append(label_node[:, leaf_mask].argmax(dim=1).detach().cpu())

            avg_train_loss = train_loss / len(train_loader)
            train_preds   = torch.cat(train_preds).numpy()
            train_targets = torch.cat(train_targets).numpy()
            train_acc = (train_preds == train_targets).mean()
            train_f1  = f1_score(train_targets, train_preds, average="macro", zero_division=0)

            self.train_loss.append(float(avg_train_loss))
            self.train_acc.append(float(train_acc))
            self.train_f1.append(float(train_f1))

            # --- Validation ---
            model.eval()
            valid_loss = 0
            valid_preds, valid_targets = [], []

            with torch.no_grad():
                for batch in valid_loader:
                    image = batch["image"].to(self.device)
                    label_node = batch["label_node"].to(self.device)

                    leaf_tensor, node_logits = model(image)
                    loss = model.loss_fn(leaf_tensor, label_node)
                    valid_loss += loss.detach().cpu().item()

                    leaf_mask = model.leaf_index.bool()
                    valid_preds.append(leaf_tensor[:, leaf_mask].argmax(dim=1).cpu())
                    valid_targets.append(label_node[:, leaf_mask].argmax(dim=1).cpu())

            avg_valid_loss = valid_loss / len(valid_loader)
            valid_preds   = torch.cat(valid_preds).numpy()
            valid_targets = torch.cat(valid_targets).numpy()
            valid_acc = (valid_preds == valid_targets).mean()
            valid_f1  = f1_score(valid_targets, valid_preds, average="macro", zero_division=0)

            self.valid_loss.append(float(avg_valid_loss))
            self.valid_acc.append(float(valid_acc))
            self.valid_f1.append(float(valid_f1))

            # --- Best model / early stopping ---
            improved = valid_acc > best_val_acc + delta
            if improved:
                best_val_acc = valid_acc
                best_epoch   = epoch + 1
                epochs_no_improve = 0
                if actual_dir is not None:
                    torch.save(model.state_dict(), best_model_path)
                    saved_flag = f" | Saved best model (val_acc={best_val_acc:.4f})"
                else:
                    saved_flag = " *"
            else:
                epochs_no_improve += 1
                saved_flag = ""

            # --- Logging ---
            if (epoch % self.print_every == 0) or (epoch == self.max_epochs - 1) or improved:
                print(
                    f"Epoch [{epoch+1}/{self.max_epochs}] "
                    f"| Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} "
                    f"| Valid Loss: {avg_valid_loss:.4f} | Valid Acc: {valid_acc:.4f} | Valid F1: {valid_f1:.4f}"
                    f"{saved_flag}"
                )

            if scheduler is not None:
                scheduler.step()

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} — no improvement for {patience} epochs.")
                break

        # --- Save training metadata ---
        if actual_dir is not None:
            training_info = {
                "hyperparameters": {
                    "learning_rate"    : self.learning_rate,
                    "max_epochs"       : self.max_epochs,
                    "gradient_clip_val": self.gradient_clip_val,
                    "patience"         : patience,
                    "delta"            : delta,
                    "scheduler"        : type(scheduler).__name__ if scheduler is not None else None,
                },
                "results": {
                    "best_epoch"        : best_epoch,
                    "best_val_acc"      : float(best_val_acc),
                    "total_epochs_run"  : len(self.train_loss),
                    "early_stopped"     : epochs_no_improve >= patience,
                },
                "training_metrics": {
                    "train_loss": self.train_loss,
                    "train_acc" : self.train_acc,
                    "train_f1"  : self.train_f1,
                    "valid_loss": self.valid_loss,
                    "valid_acc" : self.valid_acc,
                    "valid_f1"  : self.valid_f1,
                },
            }
            info_path = os.path.join(actual_dir, "training_info.json")
            with open(info_path, "w") as f:
                json.dump(training_info, f, indent=2)
            print(f"\nSaved to '{actual_dir}/'")
            print(f"  best_model.pt   — best checkpoint (epoch {best_epoch}, val_acc={best_val_acc:.4f})")
            print(f"  training_info.json — hyperparameters and per-epoch metrics")

    def predict(self, model: HierRouteNet, pred_loader: DataLoader, save: bool = False) -> dict:
        """
        Run model.predict() over a loader, call evaluate() and count_mismatches(), and return results.

        Args:
            model       : HierRouteNet
            pred_loader : DataLoader yielding batches with "image" and "label_node" keys
            save        : If True, write predictions.npz and eval_results.json to self.model_dir.
                          predictions.npz includes per-level arrays (level_{d}_preds / trues)
                          needed by Visualize.plot_pred().

        Returns:
            dict with keys: predictions, targets, pred_paths, label_node, level_results, mismatch_results
        """
        model.to(self.device)
        model.eval()

        all_pred_ids    = []
        all_true_ids    = []
        all_label_nodes = []
        all_pred_paths  = []

        with torch.no_grad():
            for batch in pred_loader:
                image      = batch["image"].to(self.device)
                label_node = batch["label_node"]

                pred_ids, pred_paths = model.predict(image)
                pred_ids = pred_ids.cpu()
                true_ids = label_node.argmax(dim=1).cpu()

                all_pred_ids.append(pred_ids)
                all_true_ids.append(true_ids)
                all_label_nodes.append(label_node.cpu())
                all_pred_paths.extend(pred_paths)

        all_pred_ids = torch.cat(all_pred_ids)
        all_true_ids = torch.cat(all_true_ids)

        level_results    = self.evaluate(model, all_pred_ids, all_true_ids, all_pred_paths)
        mismatch_results = self.count_mismatches(model, all_pred_paths)

        if save and self.model_dir is not None:
            os.makedirs(self.model_dir, exist_ok=True)
            # Per-level aligned arrays for confusion matrices in Visualize
            level_arrays = Trainer._get_level_predictions(model, all_pred_paths, all_true_ids)
            save_dict = {
                "predictions": all_pred_ids.numpy(),
                "targets"    : all_true_ids.numpy(),
            }
            for depth, (preds_arr, trues_arr) in level_arrays.items():
                save_dict[f"level_{depth}_preds"] = preds_arr
                save_dict[f"level_{depth}_trues"] = trues_arr
            np.savez(os.path.join(self.model_dir, "predictions.npz"), **save_dict)

            eval_results = {"level_results": level_results, "mismatch_results": mismatch_results}
            with open(os.path.join(self.model_dir, "eval_results.json"), "w") as f:
                json.dump(eval_results, f, indent=2)
            print(f"\nSaved to '{self.model_dir}/'")
            print(f"  predictions.npz  — leaf predictions + per-level aligned arrays")
            print(f"  eval_results.json — level and mismatch results")

        return {
            "predictions"     : all_pred_ids.numpy(),
            "targets"         : all_true_ids.numpy(),
            "pred_paths"      : all_pred_paths,
            "label_node"      : torch.cat(all_label_nodes, dim=0),
            "level_results"   : level_results,
            "mismatch_results": mismatch_results,
        }

    def evaluate(
        self,
        model: HierRouteNet,
        pred_ids: torch.Tensor,
        true_ids: torch.Tensor,
        pred_paths: list[list[int]],
    ) -> dict[int, dict]:
        """
        Compute per-level and per-class metrics using the expert paths from model.predict().

        At depth d, samples whose true leaf is shallower than d are excluded.
        The predicted node at depth d is taken directly from pred_paths[i][d];
        if the path is shorter than d, the last node in the path is used.

        Overall macro metrics are computed over the true classes only (labels present in
        y_true), so that predictions landing at the wrong depth do not introduce phantom
        zero-F1 classes that deflate the macro average.

        Args:
            model      : HierRouteNet
            pred_ids   : LongTensor (N,) — predicted leaf node_ids
            true_ids   : LongTensor (N,) — true leaf node_ids
            pred_paths : list of lists — expert paths from model.predict(), one per sample

        Returns:
            dict mapping depth (int) -> {
                "overall"  : {accuracy, f1, precision, recall, n_samples},
                "per_class": {class_name -> {node_id, accuracy, f1, precision, recall, n_samples}}
            }
        """
        true_list = true_ids.tolist()

        max_depth = max(node.depth for node in model.hierarchy.nodes.values())

        results = {}
        print("Hierarchical Evaluation:")
        for depth in range(1, max_depth + 1):
            preds_at_depth = []
            trues_at_depth = []

            for i, true_id in enumerate(true_list):
                if model.hierarchy.nodes[true_id].depth < depth:
                    continue

                true_path = model.hierarchy.get_path_to_root(true_id)
                pred_path = pred_paths[i]

                true_node = true_path[depth]
                pred_node = pred_path[depth] if depth < len(pred_path) else pred_path[-1]

                trues_at_depth.append(true_node)
                preds_at_depth.append(pred_node)

            if not trues_at_depth:
                continue

            # unique_classes is derived from true labels only — this ensures macro
            # averaging is restricted to classes that actually exist at this depth,
            # preventing predictions that land at the wrong depth from introducing
            # phantom zero-F1 classes that deflate the overall macro score.
            unique_classes = sorted(set(trues_at_depth))

            # --- Overall macro metrics (restricted to true classes) ---
            acc = sum(p == t for p, t in zip(preds_at_depth, trues_at_depth)) / len(trues_at_depth)
            prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
                trues_at_depth, preds_at_depth, labels=unique_classes, average="macro", zero_division=0
            )

            print(f"  Level {depth}: Acc={acc:.4f} | F1={f1_macro:.4f} | Prec={prec_macro:.4f} | Rec={rec_macro:.4f} | n={len(trues_at_depth)}")

            # --- Per-class metrics ---
            precs, recs, f1s, supports = precision_recall_fscore_support(
                trues_at_depth, preds_at_depth, labels=unique_classes, average=None, zero_division=0
            )

            per_class = {}
            for i, cls_id in enumerate(unique_classes):
                cls_name  = model.hierarchy.nodes[cls_id].name
                n_cls     = int(supports[i])
                correct   = sum(p == t for p, t in zip(preds_at_depth, trues_at_depth) if t == cls_id)
                cls_acc   = correct / n_cls if n_cls > 0 else 0.0
                per_class[cls_name] = {
                    "node_id"  : cls_id,
                    "accuracy" : round(cls_acc, 4),
                    "f1"       : round(float(f1s[i]), 4),
                    "precision": round(float(precs[i]), 4),
                    "recall"   : round(float(recs[i]), 4),
                    "n_samples": n_cls,
                }
                print(f"    {cls_name:<22} Acc={cls_acc:.4f} | F1={f1s[i]:.4f} | Prec={precs[i]:.4f} | Rec={recs[i]:.4f} | n={n_cls}")

            results[depth] = {
                "overall": {
                    "accuracy" : round(acc, 4),
                    "f1"       : round(float(f1_macro), 4),
                    "precision": round(float(prec_macro), 4),
                    "recall"   : round(float(rec_macro), 4),
                    "n_samples": len(trues_at_depth),
                },
                "per_class": per_class,
            }

        return results

    def count_mismatches(self, model: HierRouteNet, pred_paths: list[list[int]]) -> dict:
        """
        Checks structural consistency of predictions using the expert paths from model.predict().

        Every step in each predicted path must be a valid parent->child edge in the hierarchy.

        Args:
            model      : HierRouteNet
            pred_paths : list of lists — expert paths from model.predict(), one per sample

        Returns:
            dict with structural_errors (int), error_pct (float), and passed (bool)
        """
        structural_errors = 0

        for pred_path in pred_paths:
            for i in range(1, len(pred_path)):
                if pred_path[i] not in model.hierarchy.nodes[pred_path[i - 1]].children:
                    structural_errors += 1

        n = len(pred_paths)
        error_pct = round(100 * structural_errors / n, 2) if n > 0 else 0.0

        print("\nStructural Consistency:")
        if structural_errors == 0:
            print(f"  PASSED — all {n} predictions follow a valid path in the hierarchy")
        else:
            print(f"  FAILED — {structural_errors} / {n} predictions ({error_pct:.2f}%) contain an invalid parent->child step")

        return {"structural_errors": structural_errors, "error_pct": error_pct, "passed": structural_errors == 0}