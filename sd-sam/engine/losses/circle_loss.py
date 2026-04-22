import torch
import torch.nn as nn
from mmseg.models.builder import LOSSES

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def euclidean_sim(x, y):
    """
      Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
      Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return 1 - dist

class func_CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(func_CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m
        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss

@LOSSES.register_module()
class CircleLoss(nn.Module):
    """Circle loss class

    Parameters
    ----------
    margin : float
        Ranking loss margin
    gamma : float
    metric : string
        Distance metric (either euclidean or cosine)
    loss_weight : float
        Weight of loss.
    loss_name : str
        Name of the loss item.
    """

    def __init__(self, margin=0.25, gamma=256, metric='cosine', loss_weight=1.0, loss_name='circle_loss'):
        super(CircleLoss, self).__init__()
        self.distance_function = euclidean_sim if metric == 'euclidean' else cosine_sim
        self.metric = metric
        self.func_circle_loss = func_CircleLoss(m=margin, gamma=gamma)
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, im, s, **kwargs):
        # compute image-sentence score matrix
        # batch_size x batch_size
        scores_i2r = self.distance_function(nn.functional.normalize(im, dim=-1),
                                            nn.functional.normalize(s, dim=-1))
        scores_r2i = scores_i2r.t()
        pos = torch.eye(im.size(0))
        neg = 1 - pos
        pos = pos.bool().to(im.device)
        neg = neg.triu(diagonal=1).bool().to(im.device)
        scores_i2r = scores_i2r.reshape(-1)
        scores_r2i = scores_r2i.reshape(-1)
        # positive similarities
        # batch_size x 1
        sp1 = scores_i2r[pos.reshape(-1)]
        sp2 = scores_r2i[pos.reshape(-1)]
        # negative _matrix
        sn1 = scores_i2r[neg.reshape(-1)]
        sn2 = scores_r2i[neg.reshape(-1)]
        cost_im = self.func_circle_loss(sp1, sn1)
        cost_s = self.func_circle_loss(sp2, sn2)
        loss = (cost_s + cost_im)
        return self.loss_weight * loss

    # @property
    # def loss_name(self):
    #     """Loss Name.
    #
    #     This function must be implemented and will return the name of this
    #     loss function. This name will be used to combine different loss items
    #     by simple sum operation. In addition, if you want this loss item to be
    #     included into the backward graph, `loss_` must be the prefix of the
    #     name.
    #     Returns:
    #         str: The name of this loss item.
    #     """
    #     return self._loss_name

# if __name__ == "__main__":
#     a = torch.rand(256, 256, requires_grad=True)
#     feat1 = nn.functional.normalize(a)
#     feat2 = nn.functional.normalize(torch.rand(256, 256, requires_grad=True))
#     criterion = CircleLoss(margin=0.5, gamma=32)
#     circle_loss = criterion(feat1, feat1)
#     print(circle_loss , circle_loss*0.01)