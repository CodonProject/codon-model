from codon import *
from codon.loss.base import LossOutput, register_loss


@register_loss('prior_mutil_label')
class PriorMultiLabelSoftMarginLoss(BasicLoss):
    '''
    Loss function based on MultiLabelSoftMarginLoss that incorporates class priors.

    Formula: Adds the log of prior probabilities as a bias to the logits, then computes binary cross-entropy loss.
    For minority classes (with low prior probability), the logits are reduced by a larger value, requiring the model
    to output higher raw scores to achieve the same activation probability, thereby increasing attention to minority classes.

    Args:
        prior (list or tensor, optional): Prior probability (positive sample ratio) for each class, length equals number of classes.
                                         If not provided, num_labels must be specified to use uniform priors.
        num_labels (int, optional): Total number of classes, required when prior is None.
        tau (float): Scaling factor for prior knowledge, default is 1.0.
        reduction (str): Loss aggregation method, options: 'mean', 'sum', 'none', default is 'mean'.
    '''

    def __init__(self, prior=None, num_labels=None, tau=1.0, reduction='mean'):
        super(PriorMultiLabelSoftMarginLoss, self).__init__()
        if prior is None:
            if num_labels is None:
                raise ValueError('Either prior or num_labels must be provided.')
            prior = torch.ones(num_labels) / num_labels
        else:
            if not isinstance(prior, torch.Tensor):
                prior = torch.tensor(prior, dtype=torch.float)
                
        self.register_buffer('prior', prior)
        self.tau = tau
        self.reduction = reduction

    def forward(self, logits, targets):
        '''
        Args:
            logits (torch.Tensor): Raw model output, shape (N, C).
            targets (torch.Tensor): Multi-hot encoded labels, shape (N, C), values 0 or 1.

        Returns:
            torch.Tensor: Loss value, shape depends on reduction.
        '''
        log_prior = torch.log(self.prior)
        adjusted_logits = logits + self.tau * log_prior

        loss = F.binary_cross_entropy_with_logits(adjusted_logits, targets, reduction='none')
        if self.reduction == 'mean':
            return LossOutput(loss=loss.mean())
        elif self.reduction == 'sum':
            return LossOutput(loss=loss.sum())
        else:  # 'none'
            return LossOutput(loss=loss)


@register_loss('asymmetric_mutil_label')
class AsymmetricLoss(BasicLoss):
    '''
    Asymmetric Loss (ASL) — An asymmetric loss function for multi-label classification.

    Paper: Asymmetric Loss for Multi-Label Classification (ICCV 2021)
    Core ideas:
        - Use different focusing parameters (gamma_pos, gamma_neg) for positive and negative samples,
          preventing easy negative samples from dominating training.
        - Apply a probability clipping threshold (clip) for negative samples, discarding overly easy
          negative samples to focus the model on hard examples.

    Args:
        gamma_pos (float): Focusing parameter for positive samples, typically set to 0.
        gamma_neg (float): Focusing parameter for negative samples, typically set to 4.
        clip (float): Probability clipping threshold for negative samples; loss is set to 0 when
                      the predicted probability of a negative sample is below this value, default is 0.05.
        eps (float): Small value to prevent log of zero, default is 1e-8.
        reduction (str): Loss aggregation method, options: 'mean', 'sum', 'none', default is 'mean'.
    '''

    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8, reduction='mean'):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        '''
        Args:
            logits (torch.Tensor): Raw model output, shape (N, C).
            targets (torch.Tensor): Multi-hot encoded labels, shape (N, C), values 0 or 1.

        Returns:
            torch.Tensor: Loss value, shape depends on reduction.
        '''
        p = torch.sigmoid(logits)

        loss_pos = - (1 - p) ** self.gamma_pos * torch.log(p + self.eps) * targets

        neg_weight = (p >= self.clip).float()
        loss_neg = - (p ** self.gamma_neg) * torch.log(1 - p + self.eps) * (1 - targets) * neg_weight

        loss = loss_pos + loss_neg

        if self.reduction == 'mean':
            return LossOutput(loss=loss.mean())
        elif self.reduction == 'sum':
            return LossOutput(loss=loss.sum())
        else:
            return LossOutput(loss=loss)