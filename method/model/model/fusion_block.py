import torch
import torch.nn as nn

class AFF1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AFF1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)

        fusion_weight = self.sigmoid(self.conv1(x1) + self.conv2(x2))

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        out = fusion_weight * x1 + (1 - fusion_weight) * x2

        return out.transpose(1, 2)


class AFF2(nn.Module):
    def __init__(self, channels):
        super(AFF2, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)

        fusion_weight = self.sigmoid(self.conv1(x1) + self.conv2(x2))

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        out = fusion_weight * x1 + (1 - fusion_weight) * x2

        return out.transpose(1, 2)


class FusionBlock(nn.Module):
    def __init__(self, activation='relu', use_bn=True, in_features=20, out_features=10, num_paths=3):
        super(FusionBlock, self).__init__()
        self.num_paths = num_paths

        self.paths = nn.ModuleList([
            AFF1(in_channels=in_features, out_channels=out_features) for _ in range(num_paths)
        ])

        self.aff2_paths = nn.ModuleList([
            AFF2(channels=out_features) for _ in range(num_paths)
        ])

        self.map_layer = nn.Sequential(
            nn.Linear(in_features, in_features, bias=False),
            nn.BatchNorm1d(in_features) if use_bn else nn.Identity(),
            self.get_activation(activation),
            nn.Linear(in_features, in_features, bias=False),
            nn.BatchNorm1d(in_features) if use_bn else nn.Identity(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(out_features * num_paths, out_features, bias=False),
            nn.BatchNorm1d(out_features) if use_bn else nn.Identity(),
            self.get_activation(activation),
            nn.Linear(out_features, out_features, bias=False),
            nn.BatchNorm1d(out_features) if use_bn else nn.Identity(),
        )

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError('激活函数必须是 ReLU、Tanh 或 Sigmoid。')

    def forward(self, h):
        b, s, _ = h.shape
        h = h.contiguous().view(b * s, -1)

        mapped = self.map_layer(h)
        mapped = mapped.contiguous().view(b, s, -1)

        h = h.view(b, s, -1)

        path_outputs = [path(h, mapped) for path in self.paths]
        fused_output = torch.cat(path_outputs, dim=-1)

        fused_output = fused_output.view(b * s, -1)
        final_output = self.fusion(fused_output).view(b, s, -1)

        res2 = [aff2(path_outputs[i], final_output) for i, aff2 in enumerate(self.aff2_paths)]

        return torch.mean(torch.stack(res2, dim=0), dim=0)