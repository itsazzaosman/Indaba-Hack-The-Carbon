"""Training modules for Prithvi-based segmentation and regression models.

This module contains PyTorch Lightning modules for both segmentation and regression tasks
using the Prithvi vision transformer backbone. It provides metrics computation, training
loops, and model configuration for geospatial machine learning tasks.
"""

from typing import Any, List, Tuple

import numpy as np
import pytorch_lightning as pl
import sklearn.metrics as metrics
import torch
import torch.nn as nn

from instageo.model.model import PrithviSeg


class PrithviSegmentationModule(pl.LightningModule):
    """Prithvi Segmentation PyTorch Lightning Module."""

    def __init__(
        self,
        image_size: int = 224,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True,
        num_classes: int = 2,
        temporal_step: int = 1,
        class_weights: List[float] = [1, 2],
        ignore_index: int = -100,
        weight_decay: float = 1e-2,
        depth: int | None = None,
    ) -> None:
        """Initialization.

        Initialize the PrithviSegmentationModule, a PyTorch Lightning module for image
        segmentation.

        Args:
            image_size (int): Size of input image.
            num_classes (int): Number of classes for segmentation.
            temporal_step (int): Number of temporal steps for multi-temporal input.
            learning_rate (float): Learning rate for the optimizer.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            class_weights (List[float]): Class weights for mitigating class imbalance.
            ignore_index (int): Class index to ignore during loss computation.
            weight_decay (float): Weight decay for L2 regularization.
            depth (int | None): Number of transformer layers to use. If None, uses default
                from config.
        """
        super().__init__()
        self.net = PrithviSeg(
            image_size=image_size,
            num_classes=num_classes,
            temporal_step=temporal_step,
            freeze_backbone=freeze_backbone,
            depth=depth,
        )
        weight_tensor = torch.tensor(class_weights).float() if class_weights else None
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=weight_tensor
        )
        self.learning_rate = learning_rate
        self.ignore_index = ignore_index
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        # Initialize metric accumulators for global metrics computation
        self.reset_metric_accumulators()

    def reset_metric_accumulators(self) -> None:
        """Reset metric accumulators for a new epoch."""
        self.val_confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.long
        )
        self.test_confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.long
        )

        self.val_probabilities: List[np.ndarray] = []
        self.val_true_labels: List[np.ndarray] = []
        self.test_probabilities: List[np.ndarray] = []
        self.test_true_labels: List[np.ndarray] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor for the model.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a training step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels.long())
        self.log_metrics(outputs, labels, "train", loss)
        
        # Log current learning rate and weight decay for WandB tracking
        self.log_current_hyperparameters()
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a validation step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels.long())

        # Log only the loss per batch
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Update confusion matrix and accumulate data for ROC-AUC for global metrics computation
        self._update_confusion_matrix(outputs, labels, "val")
        self._accumulate_for_roc_auc(outputs, labels, "val")

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a test step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels.long())

        # Log only the loss per batch
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # Update confusion matrix and accumulate data for ROC-AUC for global metrics computation
        self._update_confusion_matrix(outputs, labels, "test")
        self._accumulate_for_roc_auc(outputs, labels, "test")

        return loss

    def _update_confusion_matrix(
        self, outputs: torch.Tensor, labels: torch.Tensor, stage: str
    ) -> None:
        """Update confusion matrix with current batch."""
        pred_mask = torch.argmax(outputs, dim=1)

        valid_mask = labels.ne(self.ignore_index)
        pred_valid = pred_mask[valid_mask]
        labels_valid = labels[valid_mask]

        confusion_matrix = getattr(self, f"{stage}_confusion_matrix")
        for true_class in range(self.num_classes):
            for pred_class in range(self.num_classes):
                confusion_matrix[true_class, pred_class] += torch.sum(
                    (labels_valid == true_class) & (pred_valid == pred_class)
                ).item()

    def _accumulate_for_roc_auc(
        self, outputs: torch.Tensor, labels: torch.Tensor, stage: str
    ) -> None:
        """Accumulate ONLY valid pixel probabilities and labels for ROC-AUC computation."""
        if self.num_classes > 2:
            raise NotImplementedError(
                f"ROC-AUC computation is not implemented for multiclass classification "
                f"with {self.num_classes} classes. Currently only binary classification "
                f"(num_classes=2) is supported."
            )

        valid_mask = labels.ne(self.ignore_index)
        probabilities = torch.nn.functional.softmax(outputs.detach(), dim=1)[:, 1, :, :]

        prob_valid = probabilities[valid_mask].cpu().numpy()
        labels_valid = labels[valid_mask].cpu().numpy()

        prob_list = getattr(self, f"{stage}_probabilities")
        label_list = getattr(self, f"{stage}_true_labels")

        prob_list.append(prob_valid)
        label_list.append(labels_valid)

    def on_validation_epoch_start(self) -> None:
        """Reset metric accumulators at the start of validation epoch."""
        self.val_confusion_matrix.zero_()
        self.val_probabilities.clear()
        self.val_true_labels.clear()

    def on_test_epoch_start(self) -> None:
        """Reset metric accumulators at the start of test epoch."""
        self.test_confusion_matrix.zero_()
        self.test_probabilities.clear()
        self.test_true_labels.clear()

    def on_validation_epoch_end(self) -> None:
        """Compute and log global validation metrics at the end of the epoch."""
        metrics_dict = self._compute_metrics_from_confusion_matrix(
            self.val_confusion_matrix
        )

        if self.val_probabilities and self.val_true_labels:
            all_probs = np.concatenate(self.val_probabilities)
            all_labels = np.concatenate(self.val_true_labels)
            if len(np.unique(all_labels)) > 1:
                roc_auc = metrics.roc_auc_score(all_labels, all_probs)
                metrics_dict["roc_auc"] = roc_auc

        # Log global metrics
        self.log("val_aAcc", metrics_dict["acc"], prog_bar=True, logger=True)
        if "roc_auc" in metrics_dict:
            self.log("val_roc_auc", metrics_dict["roc_auc"], prog_bar=True, logger=True)
        self.log("val_IoU", metrics_dict["iou"], prog_bar=True, logger=True)

        for idx, value in enumerate(metrics_dict["iou_per_class"]):
            self.log(f"val_IoU_{idx}", value, logger=True)
        for idx, value in enumerate(metrics_dict["acc_per_class"]):
            self.log(f"val_Acc_{idx}", value, logger=True)
        for idx, value in enumerate(metrics_dict["precision_per_class"]):
            self.log(f"val_Precision_{idx}", value, logger=True)
        for idx, value in enumerate(metrics_dict["recall_per_class"]):
            self.log(f"val_Recall_{idx}", value, logger=True)

    def on_test_epoch_end(self) -> None:
        """Compute and log global test metrics at the end of the epoch."""
        metrics_dict = self._compute_metrics_from_confusion_matrix(
            self.test_confusion_matrix
        )

        if self.test_probabilities and self.test_true_labels:
            all_probs = np.concatenate(self.test_probabilities)
            all_labels = np.concatenate(self.test_true_labels)
            if len(np.unique(all_labels)) > 1:  # Need at least 2 classes for ROC-AUC
                roc_auc = metrics.roc_auc_score(all_labels, all_probs)
                metrics_dict["roc_auc"] = roc_auc

        # Log global metrics
        self.log("test_aAcc", metrics_dict["acc"], prog_bar=True, logger=True)
        if "roc_auc" in metrics_dict:
            self.log(
                "test_roc_auc", metrics_dict["roc_auc"], prog_bar=True, logger=True
            )
        self.log("test_IoU", metrics_dict["iou"], prog_bar=True, logger=True)

        for idx, value in enumerate(metrics_dict["iou_per_class"]):
            self.log(f"test_IoU_{idx}", value, logger=True)
        for idx, value in enumerate(metrics_dict["acc_per_class"]):
            self.log(f"test_Acc_{idx}", value, logger=True)
        for idx, value in enumerate(metrics_dict["precision_per_class"]):
            self.log(f"test_Precision_{idx}", value, logger=True)
        for idx, value in enumerate(metrics_dict["recall_per_class"]):
            self.log(f"test_Recall_{idx}", value, logger=True)

    def _compute_metrics_from_confusion_matrix(
        self, confusion_matrix: torch.Tensor
    ) -> dict:
        """Compute metrics from confusion matrix."""
        cm = confusion_matrix.cpu().numpy()

        iou_per_class = []
        acc_per_class = []
        precision_per_class = []
        recall_per_class = []

        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            # IoU
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
                iou_per_class.append(iou)
            else:
                iou_per_class.append(0.0)

            # Per-class accuracy (recall)
            if cm[i, :].sum() > 0:
                acc = tp / cm[i, :].sum()
                acc_per_class.append(acc)
            else:
                acc_per_class.append(0.0)

            # Precision
            if tp + fp > 0:
                precision = tp / (tp + fp)
                precision_per_class.append(precision)
            else:
                precision_per_class.append(0.0)

            # Recall (same as per-class accuracy)
            recall_per_class.append(acc_per_class[-1])

        # Overall metrics
        overall_accuracy = cm.diagonal().sum() / cm.sum() if cm.sum() > 0 else 0.0
        mean_iou = np.mean(iou_per_class) if iou_per_class else 0.0

        return {
            "acc": overall_accuracy,
            "iou": mean_iou,
            "acc_per_class": acc_per_class,
            "iou_per_class": iou_per_class,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
        }

    def predict_step(self, batch: Any) -> torch.Tensor:
        """Perform a prediction step.

        Args:
            batch (Any): Input batch data.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        prediction = self.forward(batch)
        probabilities = torch.nn.functional.softmax(prediction, dim=1)[:, 1, :, :]
        return probabilities

    def configure_optimizers(
        self,
    ):
        """Configure the model's optimizers and learning rate schedulers.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler]]:
            A tuple containing the list of optimizers and the list of LR schedulers.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=40, eta_min=0
        )
        return [optimizer], [scheduler]

    def log_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        stage: str,
        loss: torch.Tensor,
    ) -> None:
        """Log all metrics for any stage.

        Args:
            predictions(torch.Tensor): Prediction tensor from the model.
            labels(torch.Tensor): Label mask.
            stage (str): One of train, val and test stages.
            loss (torch.Tensor): Loss value.

        Returns:
            None.
        """
        out = self.compute_metrics(predictions, labels)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_aAcc",
            out["acc"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
                logger=True,
        )
        self.log(
            f"{stage}_roc_auc",
            out["roc_auc"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_IoU",
            out["iou"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        for idx, value in enumerate(out["iou_per_class"]):
            self.log(
                f"{stage}_IoU_{idx}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
        )
        for idx, value in enumerate(out["acc_per_class"]):
            self.log(
                f"{stage}_Acc_{idx}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
        )
        for idx, value in enumerate(out["precision_per_class"]):
            self.log(
                f"{stage}_Precision_{idx}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
        )
        for idx, value in enumerate(out["recall_per_class"]):
            self.log(
                f"{stage}_Recall_{idx}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
        )

    def log_current_hyperparameters(self) -> None:
        """Log current learning rate and weight decay for WandB tracking.
        
        This method logs the current values of learning rate and weight decay
        during training, which is useful for tracking scheduler changes.
        """
        # Get current learning rate from optimizer
        optimizers = self.optimizers()
        if optimizers:
            # Handle both single optimizer and list of optimizers
            if isinstance(optimizers, list):
                if len(optimizers) > 0:
                    optimizer = optimizers[0]
                else:
                    return
            else:
                # Single optimizer case
                optimizer = optimizers
            
            if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                current_lr = optimizer.param_groups[0]['lr']
                self.log(
                    "current_learning_rate",
                    current_lr,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
        
        # Log weight decay (this typically doesn't change during training)
        self.log(
            "current_weight_decay",
            self.weight_decay,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    def compute_metrics(
        self, pred_mask: torch.Tensor, gt_mask: torch.Tensor
    ) -> dict[str, List[float]]:
        """Calculate Metrics.

        The computed metrics includes Intersection over Union (IoU), Accuracy, Precision, Recall and
        ROC-AUC.

        Args:
            pred_mask (np.array): Predicted segmentation mask.
            gt_mask (np.array): Ground truth segmentation mask.

        Returns:
            dict: A dictionary containing 'iou', 'overall_accuracy', and
                'accuracy_per_class', 'precision_per_class' and 'recall_per_class'.
        """
        prediction_proba = torch.nn.functional.softmax(pred_mask.detach(), dim=1)[
            :, 1, :, :
        ]
        pred_mask = torch.argmax(pred_mask, dim=1)
        no_ignore = gt_mask.ne(self.ignore_index).to(self.device)
        prediction_proba = prediction_proba.masked_select(no_ignore).cpu().numpy()
        pred_mask = pred_mask.masked_select(no_ignore).cpu().numpy()
        gt_mask = gt_mask.masked_select(no_ignore).cpu().numpy()
        classes = np.unique(np.concatenate((gt_mask, pred_mask)))

        iou_per_class = []
        accuracy_per_class = []
        precision_per_class = []
        recall_per_class = []

        for clas in classes:
            pred_cls = pred_mask == clas
            gt_cls = gt_mask == clas

            intersection = np.logical_and(pred_cls, gt_cls)
            union = np.logical_or(pred_cls, gt_cls)
            true_positive = np.sum(intersection)
            false_positive = np.sum(pred_cls) - true_positive
            false_negative = np.sum(gt_cls) - true_positive

            if np.any(union):
                iou = np.sum(intersection) / np.sum(union)
                iou_per_class.append(iou)

            accuracy = true_positive / np.sum(gt_cls) if np.sum(gt_cls) > 0 else 0
            accuracy_per_class.append(accuracy)

            precision = (
                true_positive / (true_positive + false_positive)
                if (true_positive + false_positive) > 0
                else 0
            )
            precision_per_class.append(precision)

            recall = (
                true_positive / (true_positive + false_negative)
                if (true_positive + false_negative) > 0
                else 0
            )
            recall_per_class.append(recall)

        # Overall IoU and accuracy
        mean_iou = np.mean(iou_per_class) if iou_per_class else 0.0
        overall_accuracy = np.sum(pred_mask == gt_mask) / gt_mask.size
        roc_auc = metrics.roc_auc_score(gt_mask, prediction_proba)
        return {
            "roc_auc": roc_auc,
            "iou": mean_iou,
            "acc": overall_accuracy,
            "acc_per_class": accuracy_per_class,
            "iou_per_class": iou_per_class,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
        }


class PrithviRegressionModule(pl.LightningModule):
    """Prithvi Regression PyTorch Lightning Module."""

    def __init__(
        self,
        image_size: int = 224,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True,
        temporal_step: int = 1,
        weight_decay: float = 1e-2,
        loss_function: str = "mse",
        ignore_index: int = -100,
        depth: int | None = None,
        log_transform: bool = False,
    ) -> None:
        """Initialization.

        Initialize the PrithviRegressionModule, a PyTorch Lightning module for regression.

        Args:
            image_size (int): Size of input image.
            learning_rate (float): Learning rate for the optimizer.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            temporal_step (int): Number of temporal steps for multi-temporal input.
            weight_decay (float): Weight decay for L2 regularization.
            loss_function (str): Loss function to use ('mse', 'mae', 'huber').
            ignore_index (int): Index value to ignore during metric computation.
            depth (int | None): Number of transformer layers to use. If None, uses default
                from config.
            log_transform (bool): Whether to apply log transformation to target values.
        """
        super().__init__()
        self.net = PrithviSeg(
            image_size=image_size,
            num_classes=1,  # Single output channel for regression
            temporal_step=temporal_step,
            freeze_backbone=freeze_backbone,
            depth=depth,
        )

        # Choose loss function
        if loss_function == "mse":
            self.criterion = nn.MSELoss(reduction="none")
        elif loss_function == "mae":
            self.criterion = nn.L1Loss(reduction="none")
        elif loss_function == "huber":
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

        self.learning_rate = learning_rate
        self.ignore_index = ignore_index
        self.weight_decay = weight_decay
        self.log_transform = log_transform

        # Initialize metric accumulators for global metrics computation
        self.reset_metric_accumulators()

    def reset_metric_accumulators(self) -> None:
        """Reset metric accumulators for a new epoch."""
        self.val_sum_squared_errors = 0.0
        self.val_sum_absolute_errors = 0.0
        self.val_sum_labels = 0.0
        self.val_sum_squared_labels = 0.0
        self.val_count = 0

        self.test_sum_squared_errors = 0.0
        self.test_sum_absolute_errors = 0.0
        self.test_sum_labels = 0.0
        self.test_sum_squared_labels = 0.0
        self.test_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor for the model.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        return self.net(x)

    def _apply_log_transform(self, labels: torch.Tensor) -> torch.Tensor:
        """Apply log transformation to labels if enabled.

        Args:
            labels (torch.Tensor): Original labels.

        Returns:
            torch.Tensor: Transformed labels.
        """
        if self.log_transform:
            # Add small epsilon to avoid log(0) and handle negative values
            epsilon = 1e-8
            labels = torch.clamp(labels, min=epsilon)
            return torch.log(labels)
        return labels

    def _inverse_log_transform(self, predictions: torch.Tensor) -> torch.Tensor:
        """Apply inverse log transformation to predictions if log transform was used.

        Args:
            predictions (torch.Tensor): Model predictions.

        Returns:
            torch.Tensor: Transformed predictions.
        """
        if self.log_transform:
            return torch.exp(predictions)
        return predictions

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a training step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        inputs, labels = batch
        outputs = self.forward(inputs)
        outputs = outputs.squeeze(1)

        valid_mask = labels.ne(self.ignore_index)
        valid_outputs = outputs[valid_mask]
        valid_labels = labels[valid_mask]

        valid_labels_transformed = self._apply_log_transform(valid_labels)

        loss = self.criterion(valid_outputs, valid_labels_transformed).mean()
        valid_outputs_original_scale = self._inverse_log_transform(valid_outputs)
        self.log_metrics(valid_outputs_original_scale, valid_labels, "train", loss)
        
        # Log current learning rate and weight decay for WandB tracking
        self.log_current_hyperparameters()
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a validation step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        inputs, labels = batch
        outputs = self.forward(inputs)
        outputs = outputs.squeeze(1)

        valid_mask = labels.ne(self.ignore_index)
        valid_outputs = outputs[valid_mask]
        valid_labels = labels[valid_mask]

        valid_labels_transformed = self._apply_log_transform(valid_labels)

        loss = self.criterion(valid_outputs, valid_labels_transformed).mean()

        # Log only the loss per batch
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        valid_outputs_original_scale = self._inverse_log_transform(valid_outputs)
        self._accumulate_regression_metrics(
            valid_outputs_original_scale, valid_labels, "val"
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a test step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        inputs, labels = batch
        outputs = self.forward(inputs)
        outputs = outputs.squeeze(1)

        valid_mask = labels.ne(self.ignore_index)
        valid_outputs = outputs[valid_mask]
        valid_labels = labels[valid_mask]

        valid_labels_transformed = self._apply_log_transform(valid_labels)

        loss = self.criterion(valid_outputs, valid_labels_transformed).mean()

        # Log only the loss per batch
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # For global metrics, use original scale and accumulate statistics
        valid_outputs_original_scale = self._inverse_log_transform(valid_outputs)
        self._accumulate_regression_metrics(
            valid_outputs_original_scale, valid_labels, "test"
        )

        return loss

    def _accumulate_regression_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor, stage: str
    ) -> None:
        """Accumulate statistics for regression metrics computation."""
        pred_np = predictions.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # Accumulate statistics
        squared_errors = (pred_np - labels_np) ** 2
        absolute_errors = np.abs(pred_np - labels_np)

        if stage == "val":
            self.val_sum_squared_errors += np.sum(squared_errors)
            self.val_sum_absolute_errors += np.sum(absolute_errors)
            self.val_sum_labels += np.sum(labels_np)
            self.val_sum_squared_labels += np.sum(labels_np**2)
            self.val_count += len(pred_np)
        else:  # test
            self.test_sum_squared_errors += np.sum(squared_errors)
            self.test_sum_absolute_errors += np.sum(absolute_errors)
            self.test_sum_labels += np.sum(labels_np)
            self.test_sum_squared_labels += np.sum(labels_np**2)
            self.test_count += len(pred_np)

    def on_validation_epoch_start(self) -> None:
        """Reset metric accumulators at the start of validation epoch."""
        self.val_sum_squared_errors = 0.0
        self.val_sum_absolute_errors = 0.0
        self.val_sum_labels = 0.0
        self.val_sum_squared_labels = 0.0
        self.val_count = 0

    def on_test_epoch_start(self) -> None:
        """Reset metric accumulators at the start of test epoch."""
        self.test_sum_squared_errors = 0.0
        self.test_sum_absolute_errors = 0.0
        self.test_sum_labels = 0.0
        self.test_sum_squared_labels = 0.0
        self.test_count = 0

    def on_validation_epoch_end(self) -> None:
        """Compute and log global validation metrics at the end of the epoch."""
        if self.val_count == 0:
            return

        # Compute global metrics from accumulated statistics
        metrics_dict = self._compute_regression_metrics_from_accumulation("val")

        self.log("val_RMSE", metrics_dict["rmse"], prog_bar=True, logger=True)
        self.log("val_MAE", metrics_dict["mae"], prog_bar=True, logger=True)
        self.log("val_R2", metrics_dict["r2"], prog_bar=True, logger=True)

    def on_test_epoch_end(self) -> None:
        """Compute and log global test metrics at the end of the epoch."""
        if self.test_count == 0:
            return

        # Compute global metrics from accumulated statistics
        metrics_dict = self._compute_regression_metrics_from_accumulation("test")

        self.log("test_RMSE", metrics_dict["rmse"], prog_bar=True, logger=True)
        self.log("test_MAE", metrics_dict["mae"], prog_bar=True, logger=True)
        self.log("test_R2", metrics_dict["r2"], prog_bar=True, logger=True)

    def _compute_regression_metrics_from_accumulation(self, stage: str) -> dict:
        """Compute regression metrics from accumulated statistics."""
        if stage == "val":
            count = self.val_count
            sum_se = self.val_sum_squared_errors
            sum_ae = self.val_sum_absolute_errors
            sum_labels = self.val_sum_labels
            sum_sq_labels = self.val_sum_squared_labels
        else:
            count = self.test_count
            sum_se = self.test_sum_squared_errors
            sum_ae = self.test_sum_absolute_errors
            sum_labels = self.test_sum_labels
            sum_sq_labels = self.test_sum_squared_labels

        if count == 0:
            return {"rmse": float("inf"), "mae": float("inf"), "r2": 0.0}

        # Calculate RMSE
        mse = sum_se / count
        rmse = np.sqrt(mse)

        # Calculate MAE
        mae = sum_ae / count

        # Calculate R² using the formula: R² = 1 - (SS_res / SS_tot)
        # SS_res = sum of squared errors (already have this)
        # SS_tot = sum of squared deviations from mean
        mean_labels = sum_labels / count
        ss_tot = sum_sq_labels - count * (
            mean_labels**2
        )  # using sum of squared deviations identity

        if ss_tot != 0:
            r2 = 1 - (sum_se / ss_tot)
        else:
            r2 = 0.0

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    def predict_step(self, batch: Any) -> torch.Tensor:
        """Perform a prediction step.

        Args:
            batch (Any): Input batch data.

        Returns:
            torch.Tensor: The predicted values in original scale.
        """
        prediction = self.forward(batch)
        prediction = prediction.squeeze(1)
        prediction = self._inverse_log_transform(prediction)

        return prediction

    def configure_optimizers(
        self,
    ):
        """Configure the model's optimizers and learning rate schedulers.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler]]:
            A tuple containing the list of optimizers and the list of LR schedulers.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=40, eta_min=0
        )
        return [optimizer], [scheduler]

    def log_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        stage: str,
        loss: torch.Tensor,
    ) -> None:
        """Log all metrics for any stage.

        Args:
            predictions(torch.Tensor): Prediction tensor from the model.
            labels(torch.Tensor): Ground truth labels.
            stage (str): One of train, val and test stages.
            loss (torch.Tensor): Loss value.

        Returns:
            None.
        """
        out = self.compute_metrics(predictions, labels)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_RMSE",
            out["rmse"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_MAE",
            out["mae"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_R2",
            out["r2"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def log_current_hyperparameters(self) -> None:
        """Log current learning rate and weight decay for WandB tracking.
        
        This method logs the current values of learning rate and weight decay
        during training, which is useful for tracking scheduler changes.
        """
        # Get current learning rate from optimizer
        optimizers = self.optimizers()
        if optimizers:
            # Handle both single optimizer and list of optimizers
            if isinstance(optimizers, list):
                if len(optimizers) > 0:
                    optimizer = optimizers[0]
                else:
                    return
            else:
                # Single optimizer case
                optimizer = optimizers
            
            if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                current_lr = optimizer.param_groups[0]['lr']
                self.log(
                    "current_learning_rate",
                    current_lr,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
        
        # Log weight decay (this typically doesn't change during training)
        self.log(
            "current_weight_decay",
            self.weight_decay,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    def compute_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float]:
        """Calculate regression metrics.

        The computed metrics include Root Mean Square Error (RMSE), Mean Absolute Error (MAE),
        and coefficient of determination (R²).

        Args:
            predictions (torch.Tensor): Predicted values.
            labels (torch.Tensor): Ground truth values.

        Returns:
            dict: A dictionary containing 'rmse', 'mae', and 'r2'.
        """
        no_ignore = labels.ne(self.ignore_index).to(self.device)
        pred_numpy = predictions.masked_select(no_ignore).detach().cpu().numpy()
        labels_numpy = labels.masked_select(no_ignore).detach().cpu().numpy()

        # Remove any remaining NaN values
        nan_mask = ~(np.isnan(pred_numpy) | np.isnan(labels_numpy))
        pred_numpy = pred_numpy[nan_mask]
        labels_numpy = labels_numpy[nan_mask]

        if len(pred_numpy) == 0:
            return {"rmse": float("inf"), "mae": float("inf"), "r2": 0.0}

        # Calculate RMSE
        mse = np.mean((pred_numpy - labels_numpy) ** 2)
        rmse = np.sqrt(mse)

        # Calculate MAE
        mae = np.mean(np.abs(pred_numpy - labels_numpy))

        # Calculate R²
        r2 = metrics.r2_score(labels_numpy, pred_numpy)

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }
