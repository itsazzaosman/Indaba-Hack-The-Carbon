### Indaba Hack the Carbon 🏆🏆 3rd Place Solution (Kaggle, hosted by InsteaDeep) 🏆🏆

This repository contains our solution that placed **3rd** in the Deep Learning Indaba Hackathon "Hack the Carbon" hosted by **InsteaDeep** on Kaggle.

> Our main approach to the solution was "bigger is better", but we also addressed several significant issues in the InstaGeo codebase.

### What this repo contains
- **Competition solution code** built on top of `InstaGeo-E2E-Geospatial-ML`.
- **Modifications to InstaGeo** to support the larger model ```Prithvi-EO-2.0-600M``` configuration and to stabilize training/inference.
- **Dataset pipeline fixes**, notably ensuring data loader workers respect configuration.
- **Experiment artifacts and configs** used during the attempts.

### Key modifications
- **Larger model support**: Adapted `InstaGeo` model and config to accommodate a bigger backbone and higher capacity settings (see `instageo/model/configs/biomass.yaml` and related training code).
- **Data pipeline fixes**: Resolved an issue where the dataset/data loader workers were effectively constrained, making the loader run with a single worker regardless of the provided configuration. Now, the configured value is honored end-to-end.
- **Training stability and logging**:
  - Improved learning rate scheduling and monitoring.
  - Reduced excessive logs; improved progress visibility.
  - Integrated config upload to experiment tracking (e.g., W&B) for reproducibility.

### How to run
1) Create and activate your environment using the provided dependencies under `InstaGeo-E2E-Geospatial-ML/requirements.txt` (or the project `pyproject.toml`).
2) Configure your run via the YAML in `instageo/model/configs/` (e.g., `biomass.yaml`).
3) Launch training or inference using the `instageo/model/run.py` entrypoints according to your setup.

### Acknowledgements
- Built on top of `InstaGeo-E2E-Geospatial-ML` and the competition starter resources.
- Thanks to the Deep Learning Indaba and InsteaDeep Kaggle organizers and community.


