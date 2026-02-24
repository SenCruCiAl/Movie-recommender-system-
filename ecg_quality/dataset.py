import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ECGDatasetConfig:
    num_leads: int = 12
    input_length: int = 2000
    sample_rate_hz: int = 500
    dataset_size: int = 512


class SyntheticECGArtifactDataset(Dataset):
    def __init__(self, config: ECGDatasetConfig):
        self.config = config
        self.rng = np.random.default_rng(42)

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = idx
        signals = self._generate_clean_signal()
        artifact_labels = np.zeros((self.config.num_leads,), dtype=np.float32)
        failure_labels = np.zeros((self.config.num_leads,), dtype=np.float32)

        for lead in range(self.config.num_leads):
            if self.rng.uniform() < 0.35:
                noise = self.rng.normal(0, 0.4, size=self.config.input_length)
                signals[lead] += noise
                artifact_labels[lead] = 1.0
            if self.rng.uniform() < 0.15:
                drift = np.linspace(0, self.rng.normal(0, 2.0), self.config.input_length)
                signals[lead] += drift
                failure_labels[lead] = 1.0
            if self.rng.uniform() < 0.05:
                signals[lead] = 0.0
                failure_labels[lead] = 1.0

        reliability_targets = 1.0 - np.clip(np.maximum(artifact_labels, failure_labels), 0.0, 1.0)

        return (
            torch.tensor(signals, dtype=torch.float32),
            torch.tensor(artifact_labels, dtype=torch.float32),
            torch.tensor(failure_labels, dtype=torch.float32),
            torch.tensor(reliability_targets, dtype=torch.float32),
        )

    def _generate_clean_signal(self) -> np.ndarray:
        t = np.arange(self.config.input_length) / self.config.sample_rate_hz
        signals = []
        for lead in range(self.config.num_leads):
            heart_rate = self.rng.uniform(55, 95)
            freq = heart_rate / 60.0
            baseline = 0.05 * np.sin(2 * math.pi * 0.33 * t)
            ecg = 0.8 * np.sin(2 * math.pi * freq * t + lead * 0.1)
            ecg += 0.2 * np.sin(2 * math.pi * 2 * freq * t)
            signals.append(ecg + baseline)
        return np.stack(signals, axis=0)


class NumpyECGArtifactDataset(Dataset):
    def __init__(self, signal_path: str, artifact_path: str, failure_path: str):
        self.signals = np.load(signal_path)
        self.artifact_labels = np.load(artifact_path)
        self.failure_labels = np.load(failure_path)
        if self.signals.shape != self.artifact_labels.shape:
            raise ValueError("Signal and artifact label arrays must have the same shape")
        if self.signals.shape != self.failure_labels.shape:
            raise ValueError("Signal and failure label arrays must have the same shape")

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        signals = self.signals[idx]
        artifact_labels = self.artifact_labels[idx]
        failure_labels = self.failure_labels[idx]
        reliability_targets = 1.0 - np.clip(
            np.maximum(artifact_labels, failure_labels), 0.0, 1.0
        )
        return (
            torch.tensor(signals, dtype=torch.float32),
            torch.tensor(artifact_labels, dtype=torch.float32),
            torch.tensor(failure_labels, dtype=torch.float32),
            torch.tensor(reliability_targets, dtype=torch.float32),
        )
