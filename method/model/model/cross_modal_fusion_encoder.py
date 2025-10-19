import torch
import torch.nn as nn
from einops import rearrange

class DualTransformerEncoderSplit(nn.Module):
    def __init__(self):
        super(DualTransformerEncoderSplit, self).__init__()

        self.eeg_encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=128,
                batch_first=True
            ) for _ in range(9)
        ])
        self.eeg_linears = nn.ModuleList([
            nn.Linear(64, 64) for _ in range(9)
        ])

        self.eye_encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                batch_first=True
            ) for _ in range(4)
        ])
        self.eye_linears = nn.ModuleList([
            nn.Linear(64, 64) for _ in range(4)
        ])

        self.eeg_fc = nn.Linear(576, 10)
        self.eye_fc = nn.Linear(256, 10)

    def forward(self, x_eeg, x_eye):
        batch, sequence, eeg_regions, eeg_features = x_eeg.shape
        batch, sequence, eye_params, eye_features = x_eye.shape

        assert eeg_regions == 9, "EEG数据的第三个维度应该是9个脑区"
        assert eye_params == 4, "眼动数据的第三个维度应该是5个参数集"

        eeg_outputs = []
        for i in range(eeg_regions):
            x_region = x_eeg[:, :, i, :]
            x_encoded = self.eeg_encoders[i](x_region)
            x_linear = self.eeg_linears[i](x_encoded)
            eeg_outputs.append(x_linear)
        x_eeg_all = torch.stack(eeg_outputs, dim=2)
        x_eeg_rearranged = rearrange(x_eeg_all, 'b s r f -> b s (r f)')
        x_eeg_reduced = self.eeg_fc(x_eeg_rearranged)

        eye_outputs = []
        for i in range(eye_params):
            x_region = x_eye[:, :, i, :]
            x_encoded = self.eye_encoders[i](x_region)
            x_linear = self.eye_linears[i](x_encoded)
            eye_outputs.append(x_linear)
        x_eye_all = torch.stack(eye_outputs, dim=2)
        x_eye_rearranged = rearrange(x_eye_all, 'b s r f -> b s (r f)')
        x_eye_reduced = self.eye_fc(x_eye_rearranged)

        return x_eeg_reduced, x_eye_reduced