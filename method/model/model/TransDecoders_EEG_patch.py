import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
N = [0, 5, 9, 6, 10, 6, 6, 9, 6, 5]


class Att_Layer(nn.Module):
    def __init__(self, input, output):
        super(Att_Layer, self).__init__()
        self.P_linear = nn.Linear(input, output, bias=True)
        self.V_linear = nn.Linear(input, 9, bias=False)

    def forward(self, att_input):
        P = self.P_linear(att_input)
        feature = torch.tanh(P)
        alpha = self.V_linear(feature)
        alpha = F.softmax(alpha, dim=2)
        out = torch.matmul(alpha, att_input)
        return out, alpha


class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(in_features=64, out_features=256)
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
        )
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=576, out_features=233)
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=233, out_features=10),
            nn.BatchNorm1d(10)
        )

    def forward(self, x):
        x = rearrange(x, 'b s l f -> b s (l f)')
        x = self.fc1(x)
        batch_size, sequence_length, features = x.shape
        x = x.view(-1, features)
        x = self.fc2(x)
        x = x.view(batch_size, sequence_length, -1)
        return x