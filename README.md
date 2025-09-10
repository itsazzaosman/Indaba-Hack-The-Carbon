### Indaba 2025 Hackathon. Hack the Carbon ([Kaggle](https://www.kaggle.com/competitions/hack-the-carbon/leaderboard), hosted by InstaDeep) 🏆🏆 3rd Place Solution 🏆🏆 

This repository contains our solution (azza Osman team) that placed **3rd** in the Deep Learning Indaba Hackathon "Hack the Carbon" hosted by **InstaDeep** on Kaggle.

> Our main approach to the solution was "bigger is better", but we also addressed several significant issues in the InstaGeo codebase.

### Problem description
Hack the Carbon is a geospatial machine learning challenge focused on estimating biomass and carbon stocks from Earth observation data. The task is a supervised regression problem where models predict biomass or a closely related proxy from multi-sensor satellite inputs (image to image regresssion).

- Input: Preprocessed geospatial chips provided by the competition baseline. a 6 bands  256 × 256 reslition chips spanning across east africa imported from Sential-2 with 3 temporal resourtions.
- Output: Predicted biomass/carbon density values for each test example in the required submission format.
- Objective: Learn spatial patterns that map remote sensing signals to biomass/carbon targets to enable scalable carbon accounting.
- Evaluation: Performance is scored on the hidden test set using the competition’s regression metric (e.g., RMSE) as defined on the leaderboard.


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
- Thanks to the Deep Learning Indaba and InstaDeep Kaggle organizers and community.


