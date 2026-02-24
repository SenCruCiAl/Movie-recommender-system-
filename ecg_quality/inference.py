import argparse
from pathlib import Path

import numpy as np
import torch

from ecg_quality.models import LeadQualityCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ECG signal quality inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = LeadQualityCNN(
        num_leads=checkpoint["num_leads"],
        input_length=checkpoint["input_length"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    signals = np.load(args.signals)
    signals_tensor = torch.tensor(signals, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(signals_tensor)

    reliability = outputs["reliability_scores"].numpy()
    artifact_prob = torch.sigmoid(outputs["artifact_logits"]).numpy()
    failure_prob = torch.sigmoid(outputs["failure_logits"]).numpy()

    print("Reliability scores per lead:\n", reliability)
    print("Artifact probabilities per lead:\n", artifact_prob)
    print("Lead failure probabilities per lead:\n", failure_prob)


if __name__ == "__main__":
    main()
