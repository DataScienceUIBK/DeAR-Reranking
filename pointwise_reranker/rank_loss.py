# rank_loss.py
import torch
import torch.nn.functional as F

class RankLoss:
    @staticmethod
    def _pairwise_diffs(y):
        # (B, N) -> pairwise differences y_i - y_j
        return y.unsqueeze(2) - y.unsqueeze(1)

    @staticmethod
    def rank_net(y_pred, y_true):
        """
        Pairwise RankNet with soft targets using sigmoid(y_true_i - y_true_j) as probability.
        y_pred, y_true: (B, N)
        """
        diff_pred = RankLoss._pairwise_diffs(y_pred)  # (B, N, N)
        diff_true = RankLoss._pairwise_diffs(y_true).detach()
        target = torch.sigmoid(diff_true)  # soft target pij
        # BCE-with-logits over upper triangle excluding diagonal
        mask = torch.triu(torch.ones_like(target), diagonal=1).bool()
        loss = F.binary_cross_entropy_with_logits(diff_pred[mask], target[mask])
        return loss

    @staticmethod
    def list_net(y_pred, y_true):
        """
        Cross-entropy between softmax of y_true and y_pred (ListNet).
        """
        p = F.log_softmax(y_pred, dim=1)
        q = F.softmax(y_true, dim=1).detach()
        return F.kl_div(p, q, reduction="batchmean")

    @staticmethod
    def _dcg(rels):
        # rels: (N,) sorted predicted order not needed here; compute ideal DCG weights externally
        idx = torch.arange(rels.size(-1), device=rels.device) + 1
        gains = (2 ** rels - 1)
        discounts = torch.log2(idx.float() + 1)
        return (gains / discounts).sum(-1)

    @staticmethod
    def lambda_loss(y_pred, y_true, k=5, weighing_scheme="ndcgLoss2_scheme"):
        """
        Approx LambdaRank: pairwise log-sigmoid weighted by |ΔNDCG|.
        """
        B, N = y_pred.shape
        device = y_pred.device

        # Ideal DCG for normalization
        ideal, _ = torch.sort(y_true, dim=1, descending=True)
        idcg = RankLoss._dcg(ideal[:, :k] if k is not None else ideal)

        # Pairwise deltas in predicted scores and true gains
        diff_pred = RankLoss._pairwise_diffs(y_pred)  # (B,N,N)
        # gains
        g = (2 ** y_true - 1.0)
        # discounts for positions are unknown a priori -> approximate with predicted order by soft ranks
        # Use a simple approx: position weights ~ 1/log2(2 + rank), where rank ~ softmax over y_pred
        # Compute ranks via sorting indices (no grad), which is standard for LambdaRank
        with torch.no_grad():
            order = torch.argsort(y_pred, dim=1, descending=True)
            pos = torch.zeros_like(order, dtype=torch.float)
            pos.scatter_(1, order, torch.arange(1, N + 1, device=device).float().unsqueeze(0).expand(B, -1))
            D = 1.0 / torch.log2(1.0 + pos + 1.0)  # (B,N)

        # ΔNDCG for swapping i,j: |(gi - gj) * (Di - Dj)| / IDCG
        Di = D.unsqueeze(2); Dj = D.unsqueeze(1)
        gi = g.unsqueeze(2); gj = g.unsqueeze(1)
        delta_dcg = torch.abs((gi - gj) * (Di - Dj))
        delta_ndcg = delta_dcg / (idcg.view(B, 1, 1) + 1e-8)

        # Only upper triangle
        mask = torch.triu(torch.ones_like(diff_pred), diagonal=1).bool()
        loss = -torch.logsigmoid(diff_pred[mask]) * delta_ndcg[mask]
        return loss.mean()
