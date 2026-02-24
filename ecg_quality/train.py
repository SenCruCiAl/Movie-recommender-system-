import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg_quality.dataset import ECGDatasetConfig, SyntheticECGArtifactDataset
from ecg_quality.models import LeadQualityCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG signal quality model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-leads", type=int, default=12)
    parser.add_argument("--input-length", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = ECGDatasetConfig(num_leads=args.num_leads, input_length=args.input_length)
    dataset = SyntheticECGArtifactDataset(config)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = LeadQualityCNN(num_leads=args.num_leads, input_length=args.input_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for signals, artifact_labels, failure_labels, reliability_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(signals)
            artifact_loss = criterion(outputs["artifact_logits"], artifact_labels)
            failure_loss = criterion(outputs["failure_logits"], failure_labels)
            reliability_loss = criterion(outputs["reliability_logits"], reliability_targets)
            loss = artifact_loss + failure_loss + 0.5 * reliability_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}: loss={running_loss / len(dataloader):.4f}")

    checkpoint = {
        "model_state": model.state_dict(),
        "num_leads": args.num_leads,
        "input_length": args.input_length,
    }
    torch.save(checkpoint, args.output / "lead_quality_cnn.pt")


if __name__ == "__main__":
    train()
