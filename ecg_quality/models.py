import torch
from torch import nn


class LeadQualityCNN(nn.Module):
    def __init__(self, num_leads: int, input_length: int):
        super().__init__()
        self.num_leads = num_leads
        self.input_length = input_length
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.artifact_head = nn.Linear(64, 1)
        self.failure_head = nn.Linear(64, 1)
        self.reliability_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() != 3:
            raise ValueError("Expected input of shape (batch, leads, length)")
        batch_size, num_leads, _ = x.shape
        if num_leads != self.num_leads:
            raise ValueError("Input lead count does not match configured model")
        x = x.reshape(batch_size * num_leads, 1, -1)
        features = self.feature_extractor(x).squeeze(-1)
        artifact_logits = self.artifact_head(features)
        failure_logits = self.failure_head(features)
        reliability_logits = self.reliability_head(features)
        outputs = {
            "artifact_logits": artifact_logits.reshape(batch_size, num_leads),
            "failure_logits": failure_logits.reshape(batch_size, num_leads),
            "reliability_logits": reliability_logits.reshape(batch_size, num_leads),
        }
        outputs["reliability_scores"] = torch.sigmoid(outputs["reliability_logits"])
        return outputs
