import torch
import torch.nn as nn

class FL_EEG(nn.Module):
    def __init__(self):
        super(FL_EEG, self).__init__()
        N = [0, 5, 9, 6, 10, 6, 6, 9, 6, 5]
        self.N = N

        self.liner_pf = nn.Linear(in_features=N[1] * 6, out_features=64)
        self.liner_f = nn.Linear(in_features=N[2] * 6, out_features=64)
        self.liner_lt = nn.Linear(in_features=N[3] * 6, out_features=64)
        self.liner_c = nn.Linear(in_features=N[4] * 6, out_features=64)
        self.liner_rt = nn.Linear(in_features=N[5] * 6, out_features=64)
        self.liner_lp = nn.Linear(in_features=N[6] * 6, out_features=64)
        self.liner_p = nn.Linear(in_features=N[7] * 6, out_features=64)
        self.liner_rp = nn.Linear(in_features=N[8] * 6, out_features=64)
        self.liner_o = nn.Linear(in_features=N[9] * 6, out_features=64)

    def map_to_64(self, x, liner):
        return liner(x.view(x.shape[0], x.shape[1], -1))

    def forward(self, x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o):
        batch, sequence, _, _ = x_pf.shape

        x_pf = self.map_to_64(x_pf, self.liner_pf)
        x_f = self.map_to_64(x_f, self.liner_f)
        x_lt = self.map_to_64(x_lt, self.liner_lt)
        x_c = self.map_to_64(x_c, self.liner_c)
        x_rt = self.map_to_64(x_rt, self.liner_rt)
        x_lp = self.map_to_64(x_lp, self.liner_lp)
        x_p = self.map_to_64(x_p, self.liner_p)
        x_rp = self.map_to_64(x_rp, self.liner_rp)
        x_o = self.map_to_64(x_o, self.liner_o)

        x_all = torch.stack([x_pf, x_f, x_lt, x_c, x_rt,
                             x_lp, x_p, x_rp, x_o], dim=2)

        return x_all


class TransformerEecoderSplit_eeg(nn.Module):
    def __init__(self):
        super(TransformerEecoderSplit_eeg, self).__init__()

        self.encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=128,
                batch_first=True
            ) for _ in range(9)
        ])

        self.linears = nn.ModuleList([
            nn.Linear(
                in_features=64,
                out_features=64
            ) for _ in range(9)
        ])

    def forward(self, x):
        batch, sequence, num_brain_regions, feature_dim = x.shape
        assert num_brain_regions == 9, "输入的第三个维度应该是9个脑区"

        outputs = []

        for i in range(num_brain_regions):
            x_region = x[:, :, i, :]
            x_encoded = self.encoders[i](x_region)
            x_linear = self.linears[i](x_encoded)
            outputs.append(x_linear)

        x_all = torch.stack(outputs, dim=2)

        return x_all