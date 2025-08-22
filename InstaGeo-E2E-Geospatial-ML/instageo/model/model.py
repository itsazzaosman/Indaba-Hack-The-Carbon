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

"""Model Module."""

import os
import time
from pathlib import Path

import numpy as np
import requests  # type: ignore
import torch
import torch.nn as nn
import json  # type: ignore
from absl import logging

from instageo.model.Prithvi import ViTEncoder, get_3d_sincos_pos_embed


def download_file(url: str, filename: str | Path, retries: int = 3) -> None:
    """Downloads a file from the given URL and saves it to a local file.

    Args:
        url (str): The URL from which to download the file.
        filename (str): The local path where the file will be saved.
        retries (int, optional): The number of times to retry the download
                                 in case of failure. Defaults to 3.

    Raises:
        Exception: If the download fails after the specified number of retries.

    Returns:
        None
    """
    if os.path.exists(filename):
        logging.info(f"File '{filename}' already exists. Skipping download.")
        return

    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                logging.info(f"Download successful on attempt {attempt + 1}")
                break
            else:
                logging.warning(
                    f"Attempt {attempt + 1} failed with status code {response.status_code}"  # noqa
                )
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed with error: {e}")

        if attempt < retries - 1:
            time.sleep(2)

    else:
        raise Exception("Failed to download the file after several attempts.")


class Norm2D(nn.Module):
    """A normalization layer for 2D inputs.

    This class implements a 2D normalization layer using Layer Normalization.
    It is designed to normalize 2D inputs (e.g., images or feature maps in a
    convolutional neural network).

    Attributes:
        ln (nn.LayerNorm): The layer normalization component.

    Args:
        embed_dim (int): The number of features of the input tensor (i.e., the number of
            channels in the case of images).

    Methods:
        forward: Applies normalization to the input tensor.
    """

    def __init__(self, embed_dim: int):
        """Initializes the Norm2D module.

        Args:
            embed_dim (int): The number of features of the input tensor.
        """
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the normalization process to the input tensor.

        Args:
            x (torch.Tensor): A 4D input tensor with shape
                (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The normalized tensor, having the same shape as the input.
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PrithviSeg(nn.Module):
    """Prithvi Segmentation Model."""

    def __init__(
        self,
        temporal_step: int = 1,
        image_size: int = 224,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        depth: int | None = None,
    ) -> None:
        """Initialize the PrithviSeg model.

        This model is designed for image segmentation tasks on remote sensing data.
        It loads Prithvi-EO-2.0-600M configuration and weights and sets up a ViTEncoder backbone
        along with a segmentation head.

        Args:
            temporal_step (int): Size of temporal dimension.
            image_size (int): Size of input image.
            num_classes (int): Number of target classes.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            depth (int | None): Number of transformer layers to use. If None, uses default
                from config.
        """
        super().__init__()
        weights_dir = Path.home() / ".instageo" / "prithvi"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "Prithvi_EO_V2_600M.pt"
        cfg_path = weights_dir / "config.json"
        download_file(
            "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M/resolve/main/Prithvi_EO_V2_600M.pt?download=true",  # noqa
            weights_path,
        )
        download_file(
            "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M/resolve/main/config.json",  # noqa
            cfg_path,
        )
        checkpoint = torch.load(weights_path, map_location="cpu")
        with open(cfg_path) as f:
            model_config = json.load(f)

        # Extract model arguments from the JSON config structure
        if "pretrained_cfg" in model_config:
            # New Prithvi-EO-2.0-600M JSON format
            pretrained_cfg = model_config["pretrained_cfg"]
            model_args = {
                "img_size": pretrained_cfg.get("img_size", 224),
                "num_frames": pretrained_cfg.get("num_frames", 4),
                "patch_size": pretrained_cfg.get("patch_size", [1, 14, 14])[1] if isinstance(pretrained_cfg.get("patch_size"), list) else pretrained_cfg.get("patch_size", 14),  # Extract spatial patch size
                "in_chans": pretrained_cfg.get("in_chans", 6),
                "embed_dim": pretrained_cfg.get("embed_dim", 1280),
                "depth": pretrained_cfg.get("depth", 32),
                "num_heads": pretrained_cfg.get("num_heads", 16),
                "mlp_ratio": pretrained_cfg.get("mlp_ratio", 4),
                "norm_pix_loss": pretrained_cfg.get("norm_pix_loss", False),
                "decoder_embed_dim": pretrained_cfg.get("decoder_embed_dim", 512),
                "decoder_depth": pretrained_cfg.get("decoder_depth", 8),
                "decoder_num_heads": pretrained_cfg.get("decoder_num_heads", 16),
            }
        elif "model_args" in model_config:
            # Legacy YAML format
            model_args = model_config["model_args"]
        else:
            # Fallback to defaults
            model_args = {}
        
        # Log the configuration for debugging
        logging.info(f"Loaded model config: {model_config}")
        logging.info(f"Extracted model args: {model_args}")
        logging.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Log key parameters for verification
        if "pretrained_cfg" in model_config:
            pretrained_cfg = model_config["pretrained_cfg"]
            logging.info(f"Prithvi-EO-2.0-600M Config: embed_dim={pretrained_cfg.get('embed_dim')}, patch_size={pretrained_cfg.get('patch_size')}, depth={pretrained_cfg.get('depth')}")

        # Override config values with user-specified parameters
        model_args["num_frames"] = temporal_step
        model_args["img_size"] = image_size
        
        if depth is not None:
            model_args["depth"] = depth
        elif "depth" not in model_args:
            # Try to infer depth from checkpoint
            max_block_idx = 0
            for key in checkpoint.keys():
                if key.startswith("encoder.blocks."):
                    try:
                        block_idx = int(key.split(".")[2])
                        max_block_idx = max(max_block_idx, block_idx)
                    except (ValueError, IndexError):
                        continue
            if max_block_idx > 0:
                model_args["depth"] = max_block_idx + 1
                logging.info(f"Extracted depth from checkpoint: {model_args['depth']}")
            else:
                model_args["depth"] = 32  # Prithvi-EO-2.0-600M default depth
            
        # Extract dimensions from checkpoint if not in config
        if "embed_dim" not in model_args:
            # Try to get embed_dim from checkpoint
            if "encoder.cls_token" in checkpoint:
                model_args["embed_dim"] = checkpoint["encoder.cls_token"].shape[-1]
                logging.info(f"Extracted embed_dim from checkpoint: {model_args['embed_dim']}")
            else:
                model_args["embed_dim"] = 1280  # Prithvi-EO-2.0-600M default
                logging.info(f"Using default embed_dim: {model_args['embed_dim']}")
                
        if "patch_size" not in model_args:
            # Try to infer patch_size from checkpoint
            if "encoder.patch_embed.proj.weight" in checkpoint:
                weight_shape = checkpoint["encoder.patch_embed.proj.weight"].shape
                # The weight shape is [out_channels, in_channels, t, h, w]
                # We can infer patch_size from the spatial dimensions
                if len(weight_shape) == 5:
                    model_args["patch_size"] = weight_shape[3]  # h dimension
                    logging.info(f"Extracted patch_size from checkpoint: {model_args['patch_size']}")
                else:
                    model_args["patch_size"] = 14  # Prithvi-EO-2.0-600M default
            else:
                model_args["patch_size"] = 14   # Prithvi-EO-2.0-600M default
                
        if "in_chans" not in model_args:
            # Try to get in_chans from checkpoint
            if "encoder.patch_embed.proj.weight" in checkpoint:
                weight_shape = checkpoint["encoder.patch_embed.proj.weight"].shape
                if len(weight_shape) == 5:
                    model_args["in_chans"] = weight_shape[1]  # in_channels dimension
                    logging.info(f"Extracted in_chans from checkpoint: {model_args['in_chans']}")
                else:
                    model_args["in_chans"] = 6  # Prithvi-EO-2.0-600M default
            else:
                model_args["in_chans"] = 6      # Prithvi-EO-2.0-600M default (6 bands)
                
        # Set default values for other parameters if not present
        if "num_heads" not in model_args:
            model_args["num_heads"] = 16  # Default number of attention heads
        if "mlp_ratio" not in model_args:
            model_args["mlp_ratio"] = 4.0  # Default MLP ratio
        if "tubelet_size" not in model_args:
            model_args["tubelet_size"] = 1  # Default tubelet size for temporal dimension
        if "norm_layer" not in model_args:
            model_args["norm_layer"] = nn.LayerNorm  # Default normalization layer
        if "norm_pix_loss" not in model_args:
            model_args["norm_pix_loss"] = False  # Default norm_pix_loss value
            
        self.model_args = model_args
        
        # Log the final model arguments for debugging
        logging.info(f"Final model args: {model_args}")
        
        # instantiate model
        try:
            model = ViTEncoder(**model_args)
            logging.info(f"Successfully created ViTEncoder with args: {model_args}")
        except Exception as e:
            logging.error(f"Failed to create ViTEncoder with args: {model_args}")
            logging.error(f"Error: {e}")
            raise
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        # Harmonize checkpoint keys with model's state_dict
        filtered_checkpoint_state_dict = {}
        encoder_keys_found = False
        
        for key, value in checkpoint.items():
            if key.startswith("encoder."):
                encoder_keys_found = True
                new_key = key[len("encoder.") :]
                # Only keep blocks from 0 to depth-1
                if new_key.startswith("blocks."):
                    try:
                        block_idx = int(new_key.split(".")[1])
                        if depth is None or block_idx < model_args.get("depth", depth or 24):
                            filtered_checkpoint_state_dict[new_key] = value
                    except (ValueError, IndexError):
                        # Skip malformed block keys
                        continue
                else:
                    filtered_checkpoint_state_dict[new_key] = value
        
        if not encoder_keys_found:
            logging.warning("No encoder keys found in checkpoint, trying direct loading")
            # If no encoder keys, try loading the checkpoint directly
            filtered_checkpoint_state_dict = checkpoint
        # Calculate patch size from image size and model configuration
        patch_size = model_args.get("patch_size", 16)  # Default to 16 if not specified
        embed_dim = model_args.get("embed_dim", 768)  # Default to 768 if not specified
        
        # Calculate the number of patches
        num_patches_h = image_size // patch_size
        num_patches_w = image_size // patch_size
        
        logging.info(f"Creating positional embedding with: embed_dim={embed_dim}, temporal_step={temporal_step}, num_patches_h={num_patches_h}, num_patches_w={num_patches_w}")
        
        filtered_checkpoint_state_dict["pos_embed"] = (
            torch.from_numpy(
                get_3d_sincos_pos_embed(
                    embed_dim,
                    (temporal_step, num_patches_h, num_patches_w),
                    cls_token=True,
                )
            )
            .float()
            .unsqueeze(0)
        )
        # Load the state dict with error handling
        try:
            missing_keys, unexpected_keys = model.load_state_dict(filtered_checkpoint_state_dict, strict=False)
            if missing_keys:
                logging.warning(f"Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
            logging.info(f"Successfully loaded checkpoint with {len(filtered_checkpoint_state_dict)} keys")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            logging.info(f"Model state dict keys: {list(model.state_dict().keys())}")
            logging.info(f"Checkpoint keys: {list(filtered_checkpoint_state_dict.keys())}")
            
            # Try to identify the specific mismatch
            model_state = model.state_dict()
            for key in filtered_checkpoint_state_dict.keys():
                if key in model_state:
                    checkpoint_shape = filtered_checkpoint_state_dict[key].shape
                    model_shape = model_state[key].shape
                    if checkpoint_shape != model_shape:
                        logging.error(f"Shape mismatch for key '{key}': checkpoint {checkpoint_shape} vs model {model_shape}")
            
            raise

        self.prithvi_600M_backbone = model
        
        # Validate that the model is working correctly
        logging.info(f"Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
        logging.info(f"Model embed_dim: {model_args['embed_dim']}, patch_size: {model_args['patch_size']}, in_chans: {model_args['in_chans']}")

        def upscaling_block(in_channels: int, out_channels: int) -> nn.Module:
            """Upscaling block.

            Args:
                in_channels (int): number of input channels.
                out_channels (int): number of output channels.

            Returns:
                An upscaling block configured to upscale spatially.
            """
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        # Calculate embedding dimensions for segmentation head
        base_embed_dim = model_args["embed_dim"]
        embed_dims = [
            (base_embed_dim * model_args["num_frames"]) // (2**i)
            for i in range(5)
        ]
        
        logging.info(f"Segmentation head embed_dims: {embed_dims}")
        
        self.segmentation_head = nn.Sequential(
            *[upscaling_block(embed_dims[i], embed_dims[i + 1]) for i in range(4)],
            nn.Conv2d(
                kernel_size=1, in_channels=embed_dims[-1], out_channels=num_classes
            ),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the model.

        Args:
            img (torch.Tensor): The input tensor representing the image.

        Returns:
            torch.Tensor: Output tensor after image segmentation.
        """
        features = self.prithvi_600M_backbone(img)
        # drop cls token
        reshaped_features = features[:, 1:, :]
        feature_img_side_length = int(
            np.sqrt(reshaped_features.shape[1] // self.model_args["num_frames"])
        )
        reshaped_features = reshaped_features.permute(0, 2, 1).reshape(
            features.shape[0], -1, feature_img_side_length, feature_img_side_length
        )

        out = self.segmentation_head(reshaped_features)
        return out
