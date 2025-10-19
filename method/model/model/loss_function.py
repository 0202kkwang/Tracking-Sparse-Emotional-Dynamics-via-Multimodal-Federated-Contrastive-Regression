import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalinterContrastiveLossModel(nn.Module):
    def __init__(self):
        super(ModalinterContrastiveLossModel, self).__init__()
        self.tau = 0.1

    def compute_modal_similarity_loss(self, local_eeg, local_eye, global_eeg, global_eye):
        cross_modal_sim1 = F.cosine_similarity(local_eeg, global_eye) / self.tau
        cross_modal_sim2 = F.cosine_similarity(local_eye, global_eeg) / self.tau
        intra_modal_sim1 = F.cosine_similarity(local_eeg, local_eye) / self.tau
        intra_modal_sim2 = F.cosine_similarity(local_eye, local_eeg) / self.tau

        modal_contrastive_loss = -torch.log(
            (torch.exp(cross_modal_sim1) + torch.exp(cross_modal_sim2)) /
            (torch.exp(intra_modal_sim1) + torch.exp(intra_modal_sim2))
        )
        return modal_contrastive_loss.mean()

    def compute_total_modal_contrastive_loss(self, local_eeg, local_eye, global_eeg, global_eye):
        batch_size, time_steps, _ = local_eeg.size()
        total_loss = 0

        for t in range(time_steps):
            local_eeg_t = local_eeg[:, t, :]
            local_eye_t = local_eye[:, t, :]
            global_eeg_t = global_eeg[:, t, :]
            global_eye_t = global_eye[:, t, :]

            step_loss = self.compute_modal_similarity_loss(
                local_eeg_t, local_eye_t, global_eeg_t, global_eye_t
            )
            total_loss += step_loss

        avg_loss = total_loss / time_steps
        return avg_loss

    def forward(self, local_eeg, local_eye, global_eeg, global_eye):
        return self.compute_total_modal_contrastive_loss(local_eeg, local_eye, global_eeg, global_eye)


class ModalintraContrastiveLossModel(nn.Module):
    def __init__(self):
        super(ModalintraContrastiveLossModel, self).__init__()
        self.tau = 0.1

    def compute_intra_modal_similarity_loss(self, local, global_, prev_local):
        pos_sim = F.cosine_similarity(local, global_) / self.tau
        neg_sim = F.cosine_similarity(local, prev_local) / self.tau

        intra_modal_loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        return intra_modal_loss.mean()

    def compute_total_intra_modal_loss(self, local_eeg, global_eeg, local_eye, global_eye):
        batch_size, time_steps, _ = local_eeg.size()
        total_eeg_loss = 0
        total_eye_loss = 0

        for t in range(1, time_steps):
            local_eeg_t = local_eeg[:, t, :]
            global_eeg_t = global_eeg[:, t, :]
            prev_local_eeg_t = local_eeg[:, t - 1, :]

            local_eye_t = local_eye[:, t, :]
            global_eye_t = global_eye[:, t, :]
            prev_local_eye_t = local_eye[:, t - 1, :]

            eeg_loss = self.compute_intra_modal_similarity_loss(
                local_eeg_t, global_eeg_t, prev_local_eeg_t
            )
            eye_loss = self.compute_intra_modal_similarity_loss(
                local_eye_t, global_eye_t, prev_local_eye_t
            )

            total_eeg_loss += eeg_loss
            total_eye_loss += eye_loss

        avg_eeg_loss = total_eeg_loss / (time_steps - 1)
        avg_eye_loss = total_eye_loss / (time_steps - 1)
        loss = avg_eeg_loss + avg_eye_loss
        return loss

    def forward(self, local_eeg, global_eeg, local_eye, global_eye):
        return self.compute_total_intra_modal_loss(local_eeg, global_eeg, local_eye, global_eye)


class BilinearCritic(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(BilinearCritic, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim1, input_dim2))
        self.fc = nn.Linear(1, hidden_dim)

    def forward(self, z1, z2):
        self.W = self.W.to(z1.device)
        z2 = z2.to(z1.device)

        score = torch.sum(z1 @ self.W * z2, dim=-1, keepdim=True)
        return self.fc(score)


class MIELossModel(nn.Module):
    def __init__(self, beta=0.3, hidden_dim=128):
        super(MIELossModel, self).__init__()
        self.beta = beta
        self.hidden_dim = hidden_dim

    def create_critic(self, input_dim1, input_dim2, device=None):
        critic = BilinearCritic(input_dim1, input_dim2, self.hidden_dim)
        if device:
            critic = critic.to(device)
        return critic

    def mutual_information_loss(self, z1, z2, critic):
        T_joint = critic(z1, z2).mean()

        z1_perm = z1[torch.randperm(z1.size(0))]
        T_marginal = torch.logsumexp(critic(z1_perm, z2), dim=0).mean()

        mie_loss = -(T_joint - T_marginal)
        return mie_loss

    def compute_mie_loss(self, eeg_all2, eye_all2,  co_feature):
        batch_size, time_steps, _ = co_feature.size()
        total_loss = 0

        for t in range(time_steps):
            eeg_feature_t = eeg_all2[:, t, :]
            eye_feature_t = eye_all2[:, t, :]
            final_feature_t = co_feature[:, t, :]

            critic1 = self.create_critic(eeg_feature_t.size(-1), final_feature_t.size(-1), device=eeg_feature_t.device)
            critic2 = self.create_critic(eye_feature_t.size(-1), final_feature_t.size(-1), device=eye_feature_t.device)

            loss_mie1 = self.mutual_information_loss(eeg_feature_t, final_feature_t, critic1)
            loss_mie2 = self.mutual_information_loss(eye_feature_t, final_feature_t, critic2)

            total_loss += (loss_mie1 + loss_mie2 )

        avg_mie_loss = total_loss / time_steps
        return avg_mie_loss