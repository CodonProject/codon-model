'''
Loss functions for image restoration tasks.

This module provides a combined loss with switchable components:
- L1 (MAE)
- L2 (MSE)
- SSIM (structural similarity)
- VGG perceptual loss
- AlexNet perceptual loss

Each component can be enabled by providing a positive weight.
'''

from codon import *
from torchvision import models
from torchvision.models import VGG19_Weights, AlexNet_Weights
from codon.loss.base import LossOutput

try:
    from pytorch_msssim import ssim as ssim_func
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


class PerceptualFeatureExtractor(BasicModel):
    '''
    Extracts intermediate features from a pretrained CNN for perceptual loss.

    Attributes:
        layers (nn.ModuleList): Selected layers of the network.
        normalize (bool): Whether to apply ImageNet normalization.
        mean (torch.Tensor): ImageNet mean.
        std (torch.Tensor): ImageNet std.
    '''

    def __init__(
        self,
        network: Literal['vgg19', 'alexnet'] = 'vgg19',
        layer_indices: Optional[list] = None,
        requires_grad: bool = False,
        use_normalize: bool = True
    ) -> None:
        '''
        Initialises the feature extractor.

        Args:
            network (Literal['vgg19', 'alexnet']): Which pretrained network to use.
            layer_indices (Optional[list]): Indices of layers to extract.
                If None, uses default: [2, 7, 12, 21] for VGG (relu1_2, relu2_2, relu3_3, relu4_3),
                and [0, 3, 6, 8, 10] for AlexNet (conv1, conv2, conv3, conv4, conv5).
            requires_grad (bool): Whether to compute gradients for the network.
            use_normalize (bool): Whether to apply ImageNet normalization to input.
        '''
        super().__init__()
        self.use_normalize = use_normalize

        # Load pretrained model
        if network == 'vgg19':
            model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            default_indices = [2, 7, 12, 21]  # relu1_2, relu2_2, relu3_3, relu4_3
        elif network == 'alexnet':
            model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features
            default_indices = [0, 3, 6, 8, 10]  # conv1, conv2, conv3, conv4, conv5
        else:
            raise ValueError(f'Unsupported network: {network}')

        self.layers = nn.ModuleList()
        indices = layer_indices if layer_indices is not None else default_indices
        for idx in indices:
            self.layers.append(model[idx])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        # ImageNet normalization constants
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Extracts features from the input.

        Args:
            x (torch.Tensor): Input tensor with values in [0, 1].

        Returns:
            torch.Tensor: Concatenated feature vectors after global adaptive pooling.
        '''
        if self.use_normalize:
            # Expect input in [0,1], normalize to ImageNet
            x = (x - self.mean) / self.std

        features = []
        for layer in self.layers:
            x = layer(x)
            # Global average pooling to fixed-size vector
            pooled = F.adaptive_avg_pool2d(x, (1, 1))
            features.append(pooled)

        return torch.cat(features, dim=1)



class ImageCombinedLoss(BasicLoss):
    '''
    Combined loss for image restoration with switchable components.

    Each component is enabled by setting its corresponding weight to a positive float.
    If weight is None (or 0.0), the component is disabled.

    Supported components:
        - L1 (MAE) loss
        - L2 (MSE) loss
        - SSIM loss (requires `pytorch-msssim`)
        - VGG perceptual loss
        - AlexNet perceptual loss
        - LPIPS loss (requires `lpips`)

    Attributes:
        weight_l1 (Optional[float]): Weight for L1 loss.
        weight_l2 (Optional[float]): Weight for L2 loss.
        weight_ssim (Optional[float]): Weight for SSIM loss.
        weight_vgg (Optional[float]): Weight for VGG perceptual loss.
        weight_alex (Optional[float]): Weight for AlexNet perceptual loss.
        weight_lpips (Optional[float]): Weight for LPIPS loss.
        input_range (str): '01' for [0,1] or '-11' for [-1,1].
        device (torch.device): Device where loss modules reside.
    '''

    def __init__(
        self,
        weight_l1: Optional[float] = None,
        weight_l2: Optional[float] = None,
        weight_ssim: Optional[float] = None,
        weight_vgg: Optional[float] = None,
        weight_alex: Optional[float] = None,
        weight_lpips: Optional[float] = None,
        input_range: Literal['01', '-11'] = '01',
        device: Optional[torch.device] = None,
        lpips_net: Literal['alex', 'vgg', 'squeeze'] = 'alex'  # LPIPS backbone
    ) -> None:
        '''
        Initialises the CombinedLoss.

        Args:
            weight_l1 (Optional[float]): Weight for L1 loss. If None, disabled.
            weight_l2 (Optional[float]): Weight for L2 loss. If None, disabled.
            weight_ssim (Optional[float]): Weight for SSIM loss. If None, disabled.
            weight_vgg (Optional[float]): Weight for VGG perceptual loss. If None, disabled.
            weight_alex (Optional[float]): Weight for AlexNet perceptual loss. If None, disabled.
            weight_lpips (Optional[float]): Weight for LPIPS loss. If None, disabled.
            input_range (Literal['01', '-11']): Range of input tensors.
                '01' means values in [0,1]; '-11' means values in [-1,1].
                Internally we convert to the range expected by LPIPS (usually [-1,1]).
            device (Optional[torch.device]): Device to place loss modules.
            lpips_net (Literal['alex', 'vgg', 'squeeze']): Backbone network for LPIPS.
        '''
        super().__init__()

        self.input_range = input_range
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Store weights
        self.weight_l1 = weight_l1 or 0.0
        self.weight_l2 = weight_l2 or 0.0
        self.weight_ssim = weight_ssim or 0.0
        self.weight_vgg = weight_vgg or 0.0
        self.weight_alex = weight_alex or 0.0
        self.weight_lpips = weight_lpips or 0.0

        # Instantiate loss modules
        self.l1_loss = nn.L1Loss(reduction='mean') if self.weight_l1 > 0 else None
        self.l2_loss = nn.MSELoss(reduction='mean') if self.weight_l2 > 0 else None

        if self.weight_ssim > 0:
            if not HAS_SSIM:
                raise ImportError('SSIM loss requires pytorch-msssim. Install with: pip install pytorch-msssim')
            self.ssim_func = ssim_func

        if self.weight_vgg > 0:
            self.vgg_extractor = PerceptualFeatureExtractor(
                network='vgg19',
                layer_indices=[2, 7, 12, 21],
                requires_grad=False,
                use_normalize=True
            ).to(device).eval()

        if self.weight_alex > 0:
            self.alex_extractor = PerceptualFeatureExtractor(
                network='alexnet',
                layer_indices=[0, 3, 6, 8, 10],
                requires_grad=False,
                use_normalize=True
            ).to(device).eval()

        if self.weight_lpips > 0:
            if not HAS_LPIPS:
                raise ImportError('LPIPS loss requires lpips. Install with: pip install lpips')
            self.lpips_model = lpips.LPIPS(net=lpips_net).to(device).eval()

        self.to(device)

    def _to_01(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_range == '-11':
            return (x + 1.0) / 2.0
        return x

    def _to_minus11(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_range == '01':
            return x * 2.0 - 1.0
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0

        pred = pred.to(self.device)
        target = target.to(self.device)

        # Pixel-level losses (expect same range as input)
        if self.l1_loss is not None:
            total_loss += self.weight_l1 * self.l1_loss(pred, target)
        if self.l2_loss is not None:
            total_loss += self.weight_l2 * self.l2_loss(pred, target)

        # SSIM (expects [0,1])
        if self.weight_ssim > 0:
            pred_ssim = self._to_01(pred)
            target_ssim = self._to_01(target)
            ssim_val = self.ssim_func(pred_ssim, target_ssim, data_range=1.0, size_average=True)
            total_loss += self.weight_ssim * (1.0 - ssim_val)

        # VGG perceptual loss (expects [0,1] normalized inside)
        if self.weight_vgg > 0:
            pred_01 = self._to_01(pred)
            target_01 = self._to_01(target)
            pred_feat = self.vgg_extractor(pred_01)
            target_feat = self.vgg_extractor(target_01)
            total_loss += self.weight_vgg * F.mse_loss(pred_feat, target_feat, reduction='mean')

        # AlexNet perceptual loss (expects [0,1])
        if self.weight_alex > 0:
            pred_01 = self._to_01(pred)
            target_01 = self._to_01(target)
            pred_feat = self.alex_extractor(pred_01)
            target_feat = self.alex_extractor(target_01)
            total_loss += self.weight_alex * F.mse_loss(pred_feat, target_feat, reduction='mean')

        # LPIPS loss (expects [-1,1])
        if self.weight_lpips > 0:
            pred_m11 = self._to_minus11(pred)
            target_m11 = self._to_minus11(target)
            # LPIPS returns a tensor of shape (batch, 1, 1, 1), we average over batch
            lpips_val = self.lpips_model(pred_m11, target_m11).mean()
            total_loss += self.weight_lpips * lpips_val

        return LossOutput(loss=total_loss)