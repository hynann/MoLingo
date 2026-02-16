import torch
import torch.nn.functional as F


# For reference
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
class KLLoss:
    def __call__(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ y_logits.T
    return sim_matrix


class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim

    def filter_sim_mat_with_sent_emb(self, sim_matrix, sent_emb):
        # put the threshold value between -1 and 1
        real_threshold_selfsim = 2 * self.threshold_selfsim - 1
        # Filtering too close values
        # mask them by putting -inf in the sim_matrix
        selfsim = sent_emb @ sent_emb.T
        selfsim_nodiag = selfsim - selfsim.diag().diag()
        idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
        sim_matrix[idx] = -torch.inf
        return sim_matrix # TODO check if return necessary or in place operation

    def __call__(self, x, y, sent_emb=None):
        bs, device = len(x), x.device
        sim_matrix = get_sim_matrix(x, y) / self.temperature

        if sent_emb is not None and self.threshold_selfsim:
            sim_matrix = self.filter_sim_mat_with_sent_emb(sim_matrix, sent_emb)

        labels = torch.arange(bs, device=device)

        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        return total_loss

    def __repr__(self):
        return f"Constrastive(temp={self.temp})"


class HN_InfoNCE_with_filtering(InfoNCE_with_filtering):
    def __init__(self, temperature=0.7, threshold_selfsim=0.8, alpha=1.0, beta=0.25):
        super().__init__(temperature=temperature, threshold_selfsim=threshold_selfsim)
        self.alpha = alpha
        self.beta = beta

    def cross_entropy_with_HN_weights(self, sim_matrix):
        n = sim_matrix.shape[0]

        labels = range(sim_matrix.shape[0])
        exp_mat = torch.exp(sim_matrix)
        num = exp_mat[range(exp_mat.shape[0]), labels]

        exp_mat_beta = torch.exp(self.beta * sim_matrix)
        weights = (n - 1) * exp_mat_beta / torch.unsqueeze((torch.sum(exp_mat_beta, axis=1) - exp_mat_beta.diag()), dim=1)
        weights = weights.fill_diagonal_(self.alpha)
        denum = torch.sum(weights * exp_mat, axis=1)

        return -torch.mean(torch.log(num/denum))

    def __call__(self, x, y, sent_emb=None):
        bs, device = len(x), x.device
        sim_matrix = get_sim_matrix(x, y) / self.temperature

        if sent_emb is not None and self.threshold_selfsim:
            sim_matrix = self.filter_sim_mat_with_sent_emb(sim_matrix, sent_emb)

        total_loss = (
            self.cross_entropy_with_HN_weights(sim_matrix) + self.cross_entropy_with_HN_weights(sim_matrix.T)
        ) / 2

        return total_loss

    def __repr__(self):
        return f"Constrastive(temp={self.temp})"
