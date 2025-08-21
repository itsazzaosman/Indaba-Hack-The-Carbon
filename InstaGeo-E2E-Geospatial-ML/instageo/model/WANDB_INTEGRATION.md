# Weights & Biases (WandB) Integration

This document explains how to use the Weights & Biases integration in InstaGeo for experiment tracking and model logging.

## Setup

1. **Install WandB**: The `wandb` package is already included in the requirements.
2. **Login to WandB**: Run `wandb login` in your terminal and follow the authentication process.

## Configuration

### Basic Usage

To enable WandB logging, modify your configuration file (e.g., `configs/config.yaml`) and set:

```yaml
train:
  use_wandb: true
  wandb_project: "your-project-name"
  wandb_run_name: "experiment-description"  # optional
  wandb_log_model: false  # set to true to log model artifacts
```

### Configuration Options

- **`use_wandb`**: Set to `true` to enable WandB logging, `false` to disable (default: `false`)
- **`wandb_project`**: The name of your WandB project (default: "instageo")
- **`wandb_run_name`**: Optional custom name for this run (default: `null`)
- **`wandb_log_model`**: Whether to log model checkpoints as artifacts (default: `false`)

### Example Configuration

See `configs/wandb_example.yaml` for a complete example with WandB enabled.

## Usage

### Command Line Override

You can also override WandB settings via command line:

```bash
python run.py train.use_wandb=true train.wandb_project="my-experiment" train.wandb_run_name="test-run-1"
```

### Running with WandB

1. **Enable WandB**: Set `use_wandb: true` in your config
2. **Run training**: Execute your training script as usual
3. **Monitor**: Check your WandB dashboard for real-time metrics

## What Gets Logged

When WandB is enabled, the following information is automatically logged:

- **Training and validation metrics**: loss, accuracy, IoU, RMSE, etc.
- **Initial hyperparameters** (logged once at start):
  - Initial learning rate
  - Initial weight decay
  - Batch size
  - Number of epochs
  - Model configuration (freeze_backbone, num_classes)
  - Data configuration (image size, temporal dimension, bands)
  - Task type (regression vs segmentation)
- **Dynamic hyperparameters** (logged every training step):
  - **Current learning rate** - tracks scheduler changes (e.g., CosineAnnealingWarmRestarts)
  - **Current weight decay** - tracks any weight decay changes
- **Training configuration**: All other settings from your config
- **Model artifacts**: Model checkpoints (if `wandb_log_model: true`)

## Integration Details

- WandB logger works alongside the existing TensorBoard logger
- No changes to existing training code are required
- WandB integration is completely optional and backward compatible
- All existing functionality remains unchanged
- **Learning rate schedulers are automatically tracked** - current LR is logged every training step
- **Weight decay changes are tracked** - current weight decay is logged every training step

## Troubleshooting

- **Authentication issues**: Ensure you're logged in with `wandb login`
- **Project not found**: Create the project in your WandB dashboard first
- **Metrics not showing**: Check that `use_wandb: true` is set in your config

## Example Workflow

1. Create a new WandB project in your dashboard
2. Copy `configs/wandb_example.yaml` and modify for your experiment
3. Set `use_wandb: true` and configure your project name
4. Run training: `python run.py --config-name wandb_example`
5. Monitor progress in your WandB dashboard
