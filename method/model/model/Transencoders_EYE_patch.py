import torch
import torch.nn as nn

class FL_EYE(nn.Module):
    def __init__(self):
        super(FL_EYE, self).__init__()
        N = [0, 12, 4, 2, 4]
        self.N = N

        self.liner_pd = nn.Linear(in_features=N[1], out_features=64)
        self.liner_di = nn.Linear(in_features=N[2], out_features=64)
        self.liner_fd = nn.Linear(in_features=N[3], out_features=64)
        self.liner_sa = nn.Linear(in_features=N[4], out_features=64)

    def map_to_64(self, x, liner):
        return liner(x)

    def forward(self, x_eye):
        batch, sequence, _ = x_eye.shape

        x_pd = self.map_to_64(x_eye[:, :, 0:12], self.liner_pd)
        x_di = self.map_to_64(x_eye[:, :, 12:16], self.liner_di)
        x_fd = self.map_to_64(x_eye[:, :, 16:18], self.liner_fd)
        x_sa = self.map_to_64(x_eye[:, :, 18:22], self.liner_sa)

        x_all = torch.stack([x_pd, x_di, x_fd, x_sa], dim=2)

        return x_all


class TransformerEecoderSplit_eye(nn.Module):
    def __init__(self):
        super(TransformerEecoderSplit_eye, self).__init__()

        self.encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                batch_first=True
            ) for _ in range(4)
        ])

        self.linears = nn.ModuleList([
            nn.Linear(
                in_features=64,
                out_features=64
            ) for _ in range(4)
        ])

    def forward(self, x):
        batch, sequence, num_brain_regions, feature_dim = x.shape
        assert num_brain_regions == 4, "输入的第三个维度应该是5个参数集"

        outputs = []

        for i in range(num_brain_regions):
            x_region = x[:, :, i, :]
            x_encoded = self.encoders[i](x_region)
            x_linear = self.linears[i](x_encoded)
            outputs.append(x_linear)

        x_all = torch.stack(outputs, dim=2)

        return x_all