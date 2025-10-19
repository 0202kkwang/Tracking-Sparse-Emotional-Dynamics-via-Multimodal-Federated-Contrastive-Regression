import torch
import torch.nn as nn
from Regression2.method.model.TEST.Transencoders_EEG_patch import TransformerEecoderSplit_eeg
from Regression2.method.model.TEST.Transencoders_EYE_patch import TransformerEecoderSplit_eye
from Regression2.method.model.TEST.Transencoders_EEG_patch import FL_EEG as FL_EEG
from Regression2.method.model.TEST.Transencoders_EYE_patch import FL_EYE as FL_EYE
from Regression2.method.model.TEST.TransDecoders_EEG_patch import Att_Layer as Att_Layer_EEG
from Regression2.method.model.TEST.TransDecoders_EEG_patch import FFN
from Regression2.method.model.TEST.TransDecoders_EEG_patch import MLP as MLP_EEG
from Regression2.method.model.TEST.TransDecoders_EYE_patch import Att_Layer as Att_Layer_EYE
from Regression2.method.model.TEST.TransDecoders_EYE_patch import MLP as MLP_EYE
from Regression2.method.model.TEST.loss_function import ModalinterContrastiveLossModel
from Regression2.method.model.TEST.loss_function import ModalintraContrastiveLossModel
from Regression2.method.model.TEST.loss_function import MIELossModel
from Regression2.method.model.TEST.cross_modal_fusion_encoder import DualTransformerEncoderSplit
from Regression2.method.model.TEST.fusion_block import FusionBlock


class concoder(nn.Module):
    def __init__(self):
        super(concoder, self).__init__()

        self.FL_EEG = FL_EEG()
        self.FL_EYE = FL_EYE()
        self.Brain_region_Att_layer = Att_Layer_EEG(input=64, output=64)
        self.EYE_feature_Att_layer = Att_Layer_EYE(input=64, output=64)
        self.MLP_EEG = MLP_EEG()
        self.MLP_EYE = MLP_EYE()
        self.FFN = FFN()
        self.TransformerEecoderSplit_eeg = TransformerEecoderSplit_eeg()
        self.TransformerEecoderSplit_eye = TransformerEecoderSplit_eye()
        self.cross_modal_encoder = DualTransformerEncoderSplit()
        self.ModalinterContrastiveLossModel = ModalinterContrastiveLossModel()
        self.ModalintraContrastiveLossModel = ModalintraContrastiveLossModel()
        self.MIELoss = MIELossModel(beta=0.3, hidden_dim=128)
        self.fusion_block = FusionBlock(in_features=20, out_features=10, num_paths=3)

        self.MHCA1 = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
        self.MHCA2 = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
        self.MHCA3 = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
        self.MHCA4 = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
        self.MHCA5 = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
        self.MHCA6 = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)

        self.fc3 = nn.Sequential(
            nn.BatchNorm1d(10),
            nn.Linear(in_features=10, out_features=40),

        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=40, out_features=20),
            nn.Linear(in_features=20, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o, x_eye):

        x_eye[torch.isnan(x_eye)] = 0

        MHCA1_outputs = []
        MHCA2_outputs = []
        MHCA3_outputs = []
        MHCA4_outputs = []
        MHCA5_outputs = []
        MHCA6_outputs = []
        all_br_weight1 = []
        all_br_weight2 = []
        all_br_weight3 = []
        all_br_weight4 = []
        all_ef_weight1 = []
        all_ef_weight2 = []
        all_ef_weight3 = []
        all_ef_weight4 = []
        eeg_out1 = []
        eeg_out2 = []
        eeg_out3 = []
        eye_out1 = []
        eye_out2 = []
        eye_out3 = []
        final_specific_eeg_out = []
        final_specific_eye_out = []

        FL_EEG = self.FL_EEG(x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o)

        eeg_encoder_layer1 = self.TransformerEecoderSplit_eeg(FL_EEG)

        eeg_encoder_layer2 = self.TransformerEecoderSplit_eeg(eeg_encoder_layer1)

        eeg_encoder_layer3 = self.TransformerEecoderSplit_eeg(eeg_encoder_layer2)

        eeg_encoder_layer4 = self.TransformerEecoderSplit_eeg(eeg_encoder_layer3)

        batch, sequence, lobe, f = eeg_encoder_layer4.shape

        for i in range(0, sequence):
            eeg, br_weight1 = self.Brain_region_Att_layer(eeg_encoder_layer4[:, i, :, :].squeeze())
            all_br_weight1.append(br_weight1)

            eeg_ffn = self.FFN(eeg.squeeze())
            eeg_out1.append(eeg_ffn)

        all_br_weight1 = torch.stack(all_br_weight1, dim=1)
        eeg_out1 = torch.stack(eeg_out1, dim=1)

        for i in range(9):
            query = eeg_out1[:, :, i, :]
            key = eeg_encoder_layer3[:, :, i, :]
            value = eeg_encoder_layer3[:, :, i, :]

            MHCA1_output, _ = self.MHCA1(query, key, value)

            MHCA1_outputs.append(MHCA1_output)

        MHCA1_outputs = torch.stack(MHCA1_outputs, dim=2)

        batch, sequence, lobe, f = MHCA1_outputs.shape

        for i in range(0, sequence):
            eeg, br_weight2 = self.Brain_region_Att_layer(MHCA1_outputs[:, i, :, :].squeeze())
            all_br_weight2.append(br_weight2)

            eeg_ffn = self.FFN(eeg.squeeze())
            eeg_out2.append(eeg_ffn)

        all_br_weight2 = torch.stack(all_br_weight2, dim=1)
        eeg_out2 = torch.stack(eeg_out2, dim=1)

        for i in range(9):
            query = eeg_out2[:, :, i, :]
            key = eeg_encoder_layer2[:, :, i, :]
            value = eeg_encoder_layer2[:, :, i, :]

            MHCA2_output, _ = self.MHCA2(query, key, value)

            MHCA2_outputs.append(MHCA2_output)

        MHCA2_outputs = torch.stack(MHCA2_outputs, dim=2)

        batch, sequence, lobe, f = MHCA2_outputs.shape

        for i in range(0, sequence):
            eeg, br_weight3 = self.Brain_region_Att_layer(MHCA2_outputs[:, i, :, :].squeeze())
            all_br_weight3.append(br_weight3)

            eeg_ffn = self.FFN(eeg.squeeze())
            eeg_out3.append(eeg_ffn)

        all_br_weight3 = torch.stack(all_br_weight3, dim=1)
        eeg_out3 = torch.stack(eeg_out3, dim=1)

        for i in range(9):
            query = eeg_out3[:, :, i, :]
            key = eeg_encoder_layer1[:, :, i, :]
            value = eeg_encoder_layer1[:, :, i, :]

            MHCA3_output, _ = self.MHCA3(query, key, value)

            MHCA3_outputs.append(MHCA3_output)

        MHCA3_outputs = torch.stack(MHCA3_outputs, dim=2)

        batch, sequence, lobe, f = MHCA3_outputs.shape

        for i in range(0, sequence):
            eeg, br_weight4 = self.Brain_region_Att_layer(MHCA3_outputs[:, i, :, :].squeeze())
            all_br_weight4.append(br_weight4)

            eeg_ffn = self.FFN(eeg.squeeze())

            final_specific_eeg_out.append(eeg_ffn)

        all_br_weight4 = torch.stack(all_br_weight4, dim=1)
        eeg_out4 = torch.stack(final_specific_eeg_out, dim=1)

        local_eeg = self.MLP_EEG(eeg_out4)

        FL_EYE = self.FL_EYE(x_eye)

        eye_encoder_layer1 = self.TransformerEecoderSplit_eye(FL_EYE)

        eye_encoder_layer2 = self.TransformerEecoderSplit_eye(eye_encoder_layer1)

        eye_encoder_layer3 = self.TransformerEecoderSplit_eye(eye_encoder_layer2)

        eye_encoder_layer4 = self.TransformerEecoderSplit_eye(eye_encoder_layer3)

        for i in range(0, sequence):
            eye, ef_weight1 = self.EYE_feature_Att_layer(eye_encoder_layer4[:, i, :, :].squeeze())
            all_ef_weight1.append(ef_weight1)

            eye_ffn = self.FFN(eye.squeeze())
            eye_out1.append(eye_ffn)

        all_ef_weight1 = torch.stack(all_ef_weight1, dim=1)
        eye_out1 = torch.stack(eye_out1, dim=1)

        for i in range(4):
            query = eye_out1[:, :, i, :]
            key = eye_encoder_layer3[:, :, i, :]
            value = eye_encoder_layer3[:, :, i, :]

            MHCA4_output, _ = self.MHCA4(query, key, value)

            MHCA4_outputs.append(MHCA4_output)

        MHCA4_outputs = torch.stack(MHCA4_outputs, dim=2)

        batch, sequence, lobe, f = MHCA4_outputs.shape

        for i in range(0, sequence):
            eye, ef_weight2 = self.EYE_feature_Att_layer(MHCA4_outputs[:, i, :, :].squeeze())
            all_ef_weight2.append(ef_weight2)

            eye_ffn = self.FFN(eye.squeeze())
            eye_out2.append(eye_ffn)

        all_ef_weight2 = torch.stack(all_ef_weight2, dim=1)
        eye_out2 = torch.stack(eye_out2, dim=1)

        for i in range(4):
            query = eye_out2[:, :, i, :]
            key = eye_encoder_layer2[:, :, i, :]
            value = eye_encoder_layer2[:, :, i, :]

            MHCA5_output, _ = self.MHCA5(query, key, value)

            MHCA5_outputs.append(MHCA5_output)

        MHCA5_outputs = torch.stack(MHCA5_outputs, dim=2)

        batch, sequence, lobe, f = MHCA5_outputs.shape

        for i in range(0, sequence):
            eye, ef_weight3 = self.EYE_feature_Att_layer(MHCA5_outputs[:, i, :, :].squeeze())
            all_ef_weight3.append(ef_weight3)

            eye_ffn = self.FFN(eye.squeeze())
            eye_out3.append(eye_ffn)

        all_ef_weight3 = torch.stack(all_ef_weight3, dim=1)
        eye_out3 = torch.stack(eye_out3, dim=1)

        for i in range(4):
            query = eye_out3[:, :, i, :]
            key = eye_encoder_layer1[:, :, i, :]
            value = eye_encoder_layer1[:, :, i, :]

            MHCA6_output, _ = self.MHCA6(query, key, value)

            MHCA6_outputs.append(MHCA6_output)

        MHCA6_outputs = torch.stack(MHCA6_outputs, dim=2)

        batch, sequence, lobe, f = MHCA6_outputs.shape

        for i in range(0, sequence):
            eye, ef_weight4 = self.EYE_feature_Att_layer(MHCA6_outputs[:, i, :, :].squeeze())
            all_ef_weight4.append(ef_weight4)

            eye_ffn = self.FFN(eye.squeeze())
            final_specific_eye_out.append(eye_ffn)

        all_br_weight4 = torch.stack(all_ef_weight4, dim=1)
        eye_out4 = torch.stack(final_specific_eye_out, dim=1)

        local_eye = self.MLP_EYE(eye_out4)

        device = torch.device('cuda:0')
        local_model1 = LocalModel1().to(device)
        local_model2 = LocalModel2().to(device)
        global_model = GlobalModel().to(device)

        eeg_all2, eye_all2 = self.cross_modal_encoder(eeg_encoder_layer4, eye_encoder_layer4)

        local_model1(eeg_all2)
        local_model2(eye_all2)

        params1 = local_model1.state_dict()
        params2 = local_model2.state_dict()

        average_params = {}
        for key in params1.keys():
            average_params[key] = (params1[key] + params2[key]) / 2

        global_model.load_state_dict(average_params)

        global_eeg = global_model(eeg_all2)
        global_eye = global_model(eye_all2)

        batch_size, sequence_length, feature_dim = local_eeg.shape

        weighted_EEG_all = torch.zeros(batch_size, sequence_length, feature_dim).to(device)
        weighted_EYE_all = torch.zeros(batch_size, sequence_length, feature_dim).to(device)

        for t in range(sequence_length):
            local_eeg_t = local_eeg[:, t, :]
            local_eye_t = local_eye[:, t, :]
            global_eeg_t = global_eeg[:, t, :]
            global_eye_t = global_eye[:, t, :]

            P_spa = global_eeg_t
            P_spe = global_eye_t

            exp_P_spa = torch.exp(P_spa)
            exp_P_spe = torch.exp(P_spe)

            gamma_spa = exp_P_spa / (exp_P_spa + exp_P_spe)
            gamma_spe = exp_P_spe / (exp_P_spa + exp_P_spe)

            weighted_EEG_t = gamma_spa * local_eeg_t
            weighted_EYE_t = gamma_spe * local_eye_t

            weighted_EEG_all[:, t, :] = weighted_EEG_t
            weighted_EYE_all[:, t, :] = weighted_EYE_t

        weighted_fusion = torch.cat((weighted_EEG_all, weighted_EYE_all), dim=2)
        fusion = self.fusion_block(weighted_fusion)
        batch, sequence, f = fusion.shape

        interContrastiveLoss = self.ModalinterContrastiveLossModel(local_eeg, local_eye, global_eeg, global_eye)
        intraContrastiveLoss = self.ModalintraContrastiveLossModel(local_eeg, global_eeg, local_eye, global_eye)
        contrastive_loss = interContrastiveLoss + intraContrastiveLoss

        mie_loss = self.MIELoss.compute_mie_loss(weighted_EEG_all, weighted_EYE_all, fusion)

        total_loss = 0.0001 * contrastive_loss + 0.01 * mie_loss

        output = []

        for i in range(0, sequence):
            x = self.fc3(fusion[:, i, :].squeeze())
            x = self.fc4(x)
            output.append(x)

        output = torch.hstack(output)

        return output, total_loss


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 3) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LocalModel1(nn.Module):
    def __init__(self):
        super(LocalModel1, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=10, num_channels=[10, 10, 10, 10, 10], kernel_size=3, dropout=0.2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        return x


class LocalModel2(nn.Module):
    def __init__(self):
        super(LocalModel2, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=10, num_channels=[10, 10, 10, 10, 10], kernel_size=3, dropout=0.2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        return x


class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=10, num_channels=[10, 10, 10, 10, 10], kernel_size=3, dropout=0.2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        return x