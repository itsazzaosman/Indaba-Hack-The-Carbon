# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""Run Module Containing Training, Evaluation and Inference Logic."""

import json
import logging
import os
from functools import partial
from typing import Callable, List, Optional, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from instageo.model.dataloader import (
    InstaGeoDataset,
    process_and_augment,
    process_data,
    process_test,
)
from instageo.model.infer_utils import chip_inference, sliding_window_inference
from instageo.model.train import PrithviRegressionModule, PrithviSegmentationModule

pl.seed_everything(seed=1042, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


def check_required_flags(required_flags: List[str], config: DictConfig) -> None:
    """Check if required flags are provided.

    Args:
        required_flags: A list of required command line arguments.

    Raises:
        An exception if at least one of the arguments is not set
    """
    for flag_name in required_flags:
        if getattr(config, flag_name) == "None":
            raise RuntimeError(f"Flag --{flag_name} is required.")


def get_device() -> str:
    """Selects available device."""
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401

        device = "tpu"
        logging.info("TPU is available. Using TPU...")
    except ImportError:
        if torch.cuda.is_available():
            device = "gpu"
            logging.info("GPU is available. Using GPU...")
        else:
            device = "cpu"
            logging.info("Neither GPU nor TPU is available. Using CPU...")
    return device


def eval_collate_fn(batch: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluation DataLoader Collate Function.

    Args:
        batch (Tuple[Tensor]): A list of tuples containing features and labels.

    Returns:
        Tuple of (x,y) concatenated into separate tensors
    """
    data = torch.cat([a[0][0] for a in batch], 0)
    labels = torch.cat([a[0][1] for a in batch], 0)
    return data, labels


def infer_collate_fn(
    batch: tuple[torch.Tensor],
) -> tuple[tuple[torch.Tensor, torch.Tensor], List[str], torch.Tensor]:
    """Inference DataLoader Collate Function.

    Args:
        batch (Tuple[Tensor]): A list of tuples containing features and labels.

    Returns:
        Tuple of (x,y) concatenated into separate tensors
    """
    data = torch.stack([a[0][0] for a in batch], 0)
    labels = [a[0][1] for a in batch]
    filepaths = [a[1] for a in batch]
    nan_mask = np.stack([(a[2]) for a in batch], 0)
    return (data, labels), filepaths, nan_mask


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 1,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for the given dataset.

    This function is a convenient wrapper around the PyTorch DataLoader class,
    allowing for easy setup of various DataLoader parameters.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
        collate_fn (Optional[Callable]): Merges a list of samples to form a mini-batch.
        pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned
            memory.

    Returns:
        DataLoader: An instance of the PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


def create_model(
    cfg: DictConfig, IM_SIZE: int, TEMPORAL_SIZE: int
) -> pl.LightningModule:
    """Create the appropriate model based on task type.

    Args:
        cfg (DictConfig): Configuration object.
        IM_SIZE (int): Image size.
        TEMPORAL_SIZE (int): Temporal dimension size.

    Returns:
        pl.LightningModule: Either PrithviSegmentationModule or PrithviRegressionModule.
    """
    if cfg.is_reg_task:
        return PrithviRegressionModule(
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            temporal_step=TEMPORAL_SIZE,
            weight_decay=cfg.train.weight_decay,
            loss_function=getattr(cfg.train, "loss_function", "mse"),
            ignore_index=cfg.train.ignore_index,
            depth=cfg.model.get("depth", None),
            log_transform=getattr(cfg.train, "log_transform", False),
        )
    else:
        return PrithviSegmentationModule(
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=TEMPORAL_SIZE,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
            weight_decay=cfg.train.weight_decay,
            depth=cfg.model.get("depth", None),
        )


def load_model_from_checkpoint(
    cfg: DictConfig, checkpoint_path: str, IM_SIZE: int, TEMPORAL_SIZE: int
) -> pl.LightningModule:
    """Load the appropriate model from checkpoint based on task type.

    Args:
        cfg (DictConfig): Configuration object.
        checkpoint_path (str): Path to the checkpoint file.
        IM_SIZE (int): Image size.
        TEMPORAL_SIZE (int): Temporal dimension size.

    Returns:
        pl.LightningModule: Either PrithviSegmentationModule or PrithviRegressionModule.
    """
    if cfg.is_reg_task:
        return PrithviRegressionModule.load_from_checkpoint(
            checkpoint_path,
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            temporal_step=TEMPORAL_SIZE,
            weight_decay=cfg.train.weight_decay,
            loss_function=getattr(cfg.train, "loss_function", "mse"),
            ignore_index=cfg.train.ignore_index,
            depth=cfg.model.get("depth", None),
            log_transform=getattr(cfg.train, "log_transform", False),
        )
    else:
        return PrithviSegmentationModule.load_from_checkpoint(
            checkpoint_path,
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=TEMPORAL_SIZE,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
            weight_decay=cfg.train.weight_decay,
            depth=cfg.model.get("depth", None),
        )


def compute_mean_std(data_loader: DataLoader) -> Tuple[List[float], List[float]]:
    """Compute the mean and standard deviation of a dataset.

    Args:
        data_loader (DataLoader): PyTorch DataLoader.

    Returns:
        mean (list): List of means for each channel.
        std (list): List of standard deviations for each channel.
    """
    mean = 0.0
    var = 0.0
    nb_samples = 0

    for data, _ in data_loader:
        # Reshape data to (B, C, T*H*W)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)

        nb_samples += batch_samples

        # Sum over batch, height and width
        mean += data.mean(2).sum(0)

        var += data.var(2, unbiased=False).sum(0)

    mean /= nb_samples
    var /= nb_samples
    std = torch.sqrt(var)
    return mean.tolist(), std.tolist()  # type:ignore


@hydra.main(config_path="configs", version_base=None, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runner Entry Point.

    Performs training, evaluation or inference/prediction depending on the selected mode.

    Arguments:
        cfg (DictConfig): Dict-like object containing necessary values used to configure runner.

    Returns:
        None.
    """
    log.info(f"Script: {__file__}")
    log.info(f"Imported hydra config:\n{OmegaConf.to_yaml(cfg)}")

    BANDS = cfg.dataloader.bands
    MEAN = cfg.dataloader.mean
    STD = cfg.dataloader.std
    IM_SIZE = cfg.dataloader.img_size
    TEMPORAL_SIZE = cfg.dataloader.temporal_dim

    batch_size = cfg.train.batch_size
    root_dir = cfg.root_dir
    valid_filepath = cfg.valid_filepath
    train_filepath = cfg.train_filepath
    test_filepath = cfg.test_filepath
    checkpoint_path = cfg.checkpoint_path

    if cfg.mode == "stats":
        train_dataset = InstaGeoDataset(
            filename=train_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=[0] * len(MEAN),
                std=[1] * len(STD),
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )
        train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        mean, std = compute_mean_std(train_loader)
        print(json.dumps({"mean": mean, "std": std}))
        exit(0)

    if cfg.mode == "train":
        check_required_flags(["root_dir", "train_filepath", "valid_filepath"], cfg)
        train_dataset = InstaGeoDataset(
            filename=train_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )

        valid_dataset = InstaGeoDataset(
            filename=valid_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
                augment=False,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )
        train_loader = create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        valid_loader = create_dataloader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        model = create_model(cfg, IM_SIZE, TEMPORAL_SIZE)
        hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        monitor = "val_RMSE" if cfg.is_reg_task else "val_IoU"
        mode = "min" if cfg.is_reg_task else "max"
        checkpoint_callback = ModelCheckpoint(
            dirpath=hydra_out_dir,
            filename="instageo_best_checkpoint",
            auto_insert_metric_name=False,
            save_top_k=1,
            monitor=monitor,
            mode=mode,
        )

        # Log learning rate every step regardless of scheduler
        lr_monitor = LearningRateMonitor(logging_interval="step")

        # Create loggers
        loggers = [TensorBoardLogger(hydra_out_dir, name="instageo")]
        
        # Add WandB logger if enabled
        if getattr(cfg.train, "use_wandb", False):
            # Log important hyperparameters to WandB
            wandb_logger = WandbLogger(
                project=getattr(cfg.train, "wandb_project", "instageo"),
                name=getattr(cfg.train, "wandb_run_name", None),
                log_model=getattr(cfg.train, "wandb_log_model", False)
            )
            
            # Log initial hyperparameters to config
            wandb_logger.experiment.config.update({
                "initial_learning_rate": cfg.train.learning_rate,
                "initial_weight_decay": cfg.train.weight_decay,
                "batch_size": cfg.train.batch_size,
                "num_epochs": cfg.train.num_epochs,
                "model_freeze_backbone": cfg.model.freeze_backbone,
                "model_num_classes": cfg.model.num_classes,
                "dataloader_img_size": cfg.dataloader.img_size,
                "dataloader_temporal_dim": cfg.dataloader.temporal_dim,
                "dataloader_bands": cfg.dataloader.bands,
                "is_regression_task": cfg.is_reg_task
            })
            
            # Store wandb_logger reference for the model to access
            # This allows the model to log current LR and weight decay during training
            wandb_logger.experiment.config["wandb_logger"] = wandb_logger
            
            loggers.append(wandb_logger)

        trainer = pl.Trainer(
            accelerator=get_device(),
            max_epochs=cfg.train.num_epochs,
            callbacks=[checkpoint_callback, lr_monitor],
            logger=loggers,
        )

        # run training and validation
        trainer.fit(model, train_loader, valid_loader)

    elif cfg.mode == "eval":
        check_required_flags(["root_dir", "test_filepath", "checkpoint_path"], cfg)
        test_dataset = InstaGeoDataset(
            filename=test_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_test,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                img_size=cfg.test.img_size,
                crop_size=cfg.test.crop_size,
                stride=cfg.test.stride,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
            include_filenames=True,
        )
        test_loader = create_dataloader(
            test_dataset, batch_size=batch_size, collate_fn=eval_collate_fn
        )
        model = load_model_from_checkpoint(cfg, checkpoint_path, IM_SIZE, TEMPORAL_SIZE)
        trainer = pl.Trainer(accelerator=get_device())
        result = trainer.test(model, dataloaders=test_loader)
        log.info(f"Evaluation results:\n{result}")

    elif cfg.mode == "sliding_inference":
        model = load_model_from_checkpoint(
            cfg, cfg.checkpoint_path, IM_SIZE, TEMPORAL_SIZE
        )
        model.eval()
        infer_filepath = os.path.join(root_dir, cfg.test_filepath)
        assert (
            os.path.splitext(infer_filepath)[-1] == ".json"
        ), f"Test file path expects a json file but got {infer_filepath}"
        output_dir = os.path.join(root_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(infer_filepath)) as json_file:
            hls_dataset = json.load(json_file)
        for key, hls_tile_path in tqdm(
            hls_dataset.items(), desc="Processing HLS Dataset"
        ):
            try:
                hls_tile, _ = process_data(
                    hls_tile_path,
                    None,
                    bands=cfg.dataloader.bands,
                    no_data_value=cfg.dataloader.no_data_value,
                    constant_multiplier=cfg.dataloader.constant_multiplier,
                    mask_cloud=cfg.test.mask_cloud,
                    replace_label=cfg.dataloader.replace_label,
                    reduce_to_zero=cfg.dataloader.reduce_to_zero,
                )
            except rasterio.RasterioIOError:
                continue
            nan_mask = hls_tile == cfg.dataloader.no_data_value
            nan_mask = np.any(nan_mask, axis=0).astype(int)
            hls_tile, _ = process_and_augment(
                hls_tile,
                None,
                mean=cfg.dataloader.mean,
                std=cfg.dataloader.std,
                temporal_size=cfg.dataloader.temporal_dim,
                augment=False,
            )
            prediction = sliding_window_inference(
                hls_tile,
                model,
                window_size=(cfg.test.img_size, cfg.test.img_size),
                stride=cfg.test.stride,
                batch_size=cfg.train.batch_size,
                device=get_device(),
            )
            prediction = np.where(nan_mask == 1, np.nan, prediction)
            prediction_filename = os.path.join(output_dir, f"{key}_prediction.tif")
            with rasterio.open(hls_tile_path["tiles"]["B02_0"]) as src:
                crs = src.crs
                transform = src.transform
            with rasterio.open(
                prediction_filename,
                "w",
                driver="GTiff",
                height=prediction.shape[0],
                width=prediction.shape[1],
                count=1,
                dtype=str(prediction.dtype),
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(prediction, 1)

    # TODO: Add support for chips that are greater than image size used for training
    elif cfg.mode == "chip_inference":
        check_required_flags(
            ["root_dir", "output_dir", "test_filepath", "checkpoint_path"], cfg
        )
        output_dir = cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)
        test_dataset = InstaGeoDataset(
            filename=test_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=cfg.test.img_size,
                augment=False,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
            include_filenames=True,
        )
        test_loader = create_dataloader(
            test_dataset, batch_size=batch_size, collate_fn=infer_collate_fn
        )
        model = load_model_from_checkpoint(cfg, checkpoint_path, IM_SIZE, TEMPORAL_SIZE)
        chip_inference(
            test_loader,
            output_dir,
            model,
            device=get_device(),
        )


if __name__ == "__main__":
    main()
