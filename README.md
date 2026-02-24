# ECG Signal Integrity ML Project

This project is a PyTorch-based pipeline for **signal quality assessment** on multi-lead ECG data. It focuses on detecting signal artifacts and lead failures and producing **per-lead reliability scores**. It intentionally avoids arrhythmia or diagnosis tasks to keep signal integrity assessment separate from clinical interpretation.

## Key capabilities

- **Multi-lead ECG ingestion** (shape: `batch x leads x length`).
- **CNN-based artifact detection** per lead.
- **Lead failure detection** per lead.
- **Reliability scores** output per lead.
- **Separation from diagnosis**: no rhythm classification is performed.

## Project structure

```
./ecg_quality
├── __init__.py
├── dataset.py
├── inference.py
├── models.py
└── train.py
```

## Usage

### 1. Train on a synthetic signal integrity dataset

```bash
python -m ecg_quality.train --epochs 5 --batch-size 32
```

This uses `SyntheticECGArtifactDataset` to generate ECG-like waveforms with injected noise, drift, and lead dropouts to simulate artifacts and failures.

### 2. Run inference

```bash
python -m ecg_quality.inference --checkpoint artifacts/lead_quality_cnn.pt --signals path/to/signals.npy
```

`signals.npy` should be shaped `[batch, leads, length]` with float values. The script prints:
- Per-lead reliability scores
- Per-lead artifact probabilities
- Per-lead lead failure probabilities

## Data formats

For real datasets, load arrays shaped `[samples, leads, length]` and labels shaped the same:

- `signals.npy`: raw ECG signals
- `artifact_labels.npy`: binary labels indicating noise/contamination per lead
- `failure_labels.npy`: binary labels indicating lead dropouts or saturation

The `NumpyECGArtifactDataset` class in `dataset.py` can be used for this data.

## Modeling notes

- The CNN extracts per-lead features, then outputs three heads:
  - `artifact_logits`: detects contamination
  - `failure_logits`: detects lead failure
  - `reliability_scores`: sigmoid output for signal reliability
- Reliability is trained as a separate head to keep quality assessment distinct from any diagnostic modeling.

## Requirements

- Python 3.9+
- PyTorch
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```
