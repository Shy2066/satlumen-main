# SatLumen Main Experiment

This repository contains only SatLumen's main experiments on the clean Fiber and Starlink datasets: input processing, the directional-count representation, training, source-domain validation for checkpoint selection, and evaluation in four directions. It uses three random seeds (42, 2024, and 2025), for a total of six training runs and twelve evaluations. Other attack methods are not included.

## Requirements

- Python 3.10
- pip
- For GPU experiments, an NVIDIA driver compatible with CUDA 11.8. The original paper experiments used an RTX 3090.

The dependency versions are pinned in `requirements.txt` based on checks against the local environment. They have not yet been validated by installing them into a fresh environment.

Create and activate a virtual environment, then install the dependencies from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Repository Structure

```text
configs/          Main experiment configuration
data_process/    Trace conversion, fixed splits, and count-cache preparation
src/              Main model, data loading, trainer, and utilities
scripts/          Main workflow, single-task training, and evaluation
```

`configs/default.yaml` is the only experiment configuration. It keeps shared model and training settings at the top level and defines `validation_candidate` and `formal_frozen` profiles for the two workflow stages. Training and cache preparation select the candidate profile; evaluation selects the frozen profile, which alone exposes `test_file`. Paths are resolved relative to the repository root, so run commands from that directory.

## Data Source and Preparation

The data and model weights are not included. This project uses only the clean Tor-over-Fiber and Tor-over-Starlink traces from the original dataset. The source is `WFdata75x80.zip`, version v2, in the [Zenodo dataset record](https://zenodo.org/records/10641853) (DOI: [10.5281/zenodo.10641853](https://doi.org/10.5281/zenodo.10641853)). The archive MD5 listed by the record is `689527862a162a1a5a448d05038e7a1b`.

The dataset is associated with the paper [Connecting the Dots in the Sky: Website Fingerprinting in Low Earth Orbit Satellite Internet](https://doi.org/10.14722/spacesec.2024.23015). This project does not redistribute the original traces. Downloading, using, or redistributing the dataset is subject to the rights, license, and conditions specified by the Zenodo record and the original data providers. If the record does not clearly state a license, do not assume redistribution is permitted.

Place pre-generated split files in this layout:

```text
npz_dataset/
├── tor_fiber/
│   ├── train.npz
│   ├── valid.npz
│   └── test.npz
└── tor_starlink/
    ├── train.npz
    ├── valid.npz
    └── test.npz
```

Each NPZ file contains `X`, `y`, and `sample_ids`. `X` has shape `(N, 10000, 2)` and stores padded events with relative timestamps and signed direction markers. `y` contains class labels from 0 to 74. The train, validation, and test splits contain 4,860, 540, and 600 samples, respectively. Each visit has the same split role in both domains.

To generate count caches from existing fixed splits, run:

```bash
python data_process/prepare_datasets.py --skip-convert --skip-split --splits train valid test
```

The count cache and main workflow record hashes of their inputs.

To start from raw text traces, place the files in `dataset/tor_fiber/` and `dataset/tor_starlink/`. Name each file `<class>-<visit-number>` and use two tab-separated columns per line. Review the plan first, then run conversion and splitting:

```bash
python data_process/prepare_datasets.py --splits train valid test --dry-run
python data_process/prepare_datasets.py --splits train valid test
```

For reproducing the paper results, use the same fixed splits and sample order.

## Running the Experiments

Preview the six training runs and twelve evaluations without starting training:

```bash
python scripts/run_main_experiment.py --dry-run
```

Run the complete main experiment:

```bash
python scripts/run_main_experiment.py --gpus 0
```

The workflow first trains models for both domains and all three seeds. Each model is selected using source-domain validation Macro-F1, then evaluated on Fiber-to-Fiber, Starlink-to-Starlink, Fiber-to-Starlink, and Starlink-to-Fiber. A run ID is generated automatically. Outputs are written to `logs/`, `checkpoints/`, and `results/`. To resume a run, use its original run ID from the log:

```bash
python scripts/run_main_experiment.py --gpus 0 --resume RUN_ID
```

Examples for running an individual training or evaluation task:

```bash
python scripts/train.py --dataset tor_fiber --seed 42 --run-id manual_fiber42
python scripts/test.py \
  --config configs/default.yaml \
  --dataset tor_starlink --source-dataset tor_fiber --seed 42 \
  --checkpoint checkpoints/directional_count/candidate/manual_fiber42/tor_fiber/seed_42/best_model.pth \
  --output results/manual_fiber42/F_to_S_seed_42.json
```

The model input covers 120 seconds and uses directional counts with shape `2 x 2700`; the first 5,000 valid events are used. Training runs for 100 epochs with a batch size of 64. Outputs include per-sample metrics, predictions, and labels. The `F1-score` field is Macro-F1, and metrics use a 0–100 scale. Newly generated single-evaluation JSON files retain full-precision metrics. For summaries, scores are recomputed from the `labels` and `predictions` arrays in each `.predictions.npz` file, including when resuming older runs. The mean and sample standard deviation (`ddof=1`) are computed across the three seeds, and values are rounded to two decimal places only at the end, matching the precision of the paper tables.

## Licensing

The code is licensed under the MIT License; see [`LICENSE`](LICENSE). Dataset rights and conditions are separate; see [`LICENSE_NOTICE.md`](LICENSE_NOTICE.md) and the original Zenodo record.
