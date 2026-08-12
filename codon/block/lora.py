from codon import *


class BasicLoRA(BasicModel):
    '''
    Base class for Low-Rank Adaptation (LoRA) modules.

    Attributes:
        original_layer (nn.Module): The original layer to be adapted.
        r (int): The rank of LoRA.
        lora_alpha (int): The scaling factor for LoRA.
        scaling (float): Actual scaling ratio (lora_alpha / r).
        merged (bool): Whether the LoRA weights are merged into the original weights.
        gate (bool): Whether to use Gated LoRA.
        lora_gate (nn.Parameter): Gate parameter.
        dora (bool): Whether to use DoRA.
        dora_m (nn.Parameter): Magnitude vector for DoRA.
        lora_a (nn.Module): Dimension reduction component.
        lora_b (nn.Module): Dimension expansion component.
        lora_dropout (nn.Module): Dropout layer for LoRA path.
    '''

    def __init__(
        self,
        original_layer: nn.Module,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
        gate: bool = False,
        dora: bool = False,
        gradient_checkpointing: bool = False
    ) -> None:
        '''
        Initializes the BasicLoRA module.

        Args:
            original_layer (nn.Module): The original layer to be adapted.
            r (int): The rank of the low-rank adaptation.
            lora_alpha (int): The scaling factor for LoRA.
            lora_dropout (float): Dropout probability for LoRA path.
            merge_weights (bool): Whether to merge LoRA weights upon initialization.
            gate (bool): Whether to use Gated LoRA.
            dora (bool): Whether to use DoRA.
            gradient_checkpointing (bool): Whether to use gradient checkpointing.
        '''
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.original_layer = original_layer
        self.lora_dropout_p = lora_dropout
        self.merge_weights = merge_weights

        # Freeze original layer
        for p in self.original_layer.parameters():
            p.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 0
        self.merged = False
        self.gate = gate
        self.dora = dora

        self.lora_gate = None
        self.lora_a = None
        self.lora_b = None
        self.lora_dropout = None
        self.dora_m = None
        self.register_buffer('weight_backup', None, persistent=False)

    @property
    def weight(self) -> Any:
        '''Exposes original layer weight for external access.'''
        return getattr(self.original_layer, 'weight', None)

    @property
    def bias(self) -> Any:
        '''Exposes original layer bias for external access.'''
        return getattr(self.original_layer, 'bias', None)

    def __getattr__(self, name: str) -> Any:
        '''Delegates attribute lookup to the original layer if not found.'''
        try:
            return super().__getattr__(name)
        except AttributeError:
            if '_modules' in self.__dict__ and 'original_layer' in self._modules:
                return getattr(self._modules['original_layer'], name)
            raise

    def reset_parameters(self) -> None:
        '''
        Resets LoRA parameters. Should be implemented by subclasses.
        '''
        pass

    def merge(self) -> None:
        '''
        Merges LoRA weights into the original layer weights. Should be implemented by subclasses.
        '''
        pass

    def unmerge(self) -> None:
        '''
        Subtracts LoRA weights from the original weights. Should be implemented by subclasses.
        '''
        pass

    def train(self, mode: bool = True) -> None:
        '''
        Sets the module in training mode and ensures weights are unmerged.

        Args:
            mode (bool): Whether to set to training mode.
        '''
        super().train(mode)
        if mode and self.merged:
            self.unmerge()

    def _forward_impl(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Internal implementation of the forward pass. Should be implemented by subclasses.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        raise NotImplementedError

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call with gradient checkpointing support.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        if self.gradient_checkpointing and self.training:
            if input_tensor.requires_grad:
                return self.checkpoint(self._forward_impl, input_tensor)
            else:
                dummy = torch.tensor(0.0, requires_grad=True, device=input_tensor.device)
                return self.checkpoint(lambda d, x: self._forward_impl(x), dummy, input_tensor)
        return self._forward_impl(input_tensor)


class LinearLoRA(BasicLoRA):
    '''
    Implements Low-Rank Adaptation (LoRA) for linear layers.

    Formula: h = W_0 x + B A x * scaling

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
    '''

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
        gate: bool = False,
        dora: bool = False,
        gradient_checkpointing: bool = False
    ) -> None:
        '''
        Initializes the LinearLoRA module.

        Args:
            original_layer (nn.Linear): The original linear layer to be adapted.
            r (int): The rank of the low-rank adaptation. Defaults to 8.
            lora_alpha (int): The scaling factor for LoRA. Defaults to 16.
            lora_dropout (float): Dropout probability for LoRA path. Defaults to 0.05.
            merge_weights (bool): Whether to merge LoRA weights upon initialization. Defaults to False.
            gate (bool): Whether to use Gated LoRA. Defaults to False.
            dora (bool): Whether to use DoRA. Defaults to False.
            gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
        '''
        super().__init__(
            original_layer, r, lora_alpha, lora_dropout,
            merge_weights, gate, dora, gradient_checkpointing
        )

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        if r > 0:
            self.lora_gate = nn.Parameter(torch.tensor([1.0])) if gate else None
            self.lora_a = nn.Parameter(torch.zeros((r, self.in_features)))
            self.lora_b = nn.Parameter(torch.zeros((self.out_features, r)))
            self.lora_dropout = nn.Dropout(p=lora_dropout)

        self.reset_parameters()

        if dora and r > 0:
            self.dora_m = nn.Parameter(self.original_layer.weight.norm(p=2, dim=1, keepdim=True))

        # Ensure LoRA parameters are on the same device as the original layer
        if hasattr(self.original_layer, 'weight'):
            self.to(self.original_layer.weight.device)

        if merge_weights:
            self.merge()

    def reset_parameters(self) -> None:
        '''
        Resets LoRA parameters.
        
        Matrix A is initialized using Kaiming Uniform initialization, 
        and matrix B is initialized to zero. This ensures the LoRA branch 
        initially has no impact on the output.
        '''
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def merge(self) -> None:
        '''
        Merges LoRA weights into the original layer weights.
        
        Used for inference acceleration.
        Note: For DoRA, merging is destructive as original weights cannot be 
        exactly recovered without a copy.
        '''
        if self.r > 0 and not self.merged:
            if self.dora:
                if self.weight_backup is None:
                    self.weight_backup = self.original_layer.weight.data.clone()

                # Calculate full weight W' = W0 + BA * scaling
                delta_w = (self.lora_b @ self.lora_a) * self.scaling
                if self.gate:
                    delta_w *= self.lora_gate
                weight = self.original_layer.weight + delta_w
                
                # Normalize and scale: W_final = m * W' / ||W'||
                norm = weight.norm(p=2, dim=1, keepdim=True)
                weight = (weight / (norm + 1e-6)) * self.dora_m
                
                # Update original weight (Destructive!)
                self.original_layer.weight.data = weight.to(self.original_layer.weight.dtype)
            else:
                # W_new = W_old + B @ A * scaling
                delta_w = (self.lora_b @ self.lora_a) * self.scaling
                if self.gate:
                    delta_w *= self.lora_gate
                self.original_layer.weight.data += delta_w.to(self.original_layer.weight.dtype)
            
            self.merged = True

    def unmerge(self) -> None:
        '''
        Subtracts LoRA weights from the original weights.
        
        Used to restore original weights or continue training.
        '''
        if self.r > 0 and self.merged:
            if self.dora:
                if self.weight_backup is not None:
                    self.original_layer.weight.data.copy_(self.weight_backup)
                    self.weight_backup = None
                else:
                    print('Warning: DoRA weights cannot be unmerged as no backup was found.')
            else:
                delta_w = (self.lora_b @ self.lora_a) * self.scaling
                if self.gate:
                    delta_w *= self.lora_gate
                self.original_layer.weight.data -= delta_w
            
            self.merged = False

    def _forward_impl(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Internal implementation of the forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        if self.r > 0 and self.merged:
            return self.original_layer(input_tensor)
        
        if self.dora and self.r > 0:
            # DoRA: W_final = m * (W0 + BA) / ||W0 + BA||
            delta_w = (self.lora_b @ self.lora_a) * self.scaling
            if self.gate:
                delta_w *= self.lora_gate
            
            # Reconstruct full weight for calculation
            weight = self.original_layer.weight + delta_w
            norm = weight.norm(p=2, dim=1, keepdim=True)
            weight = (weight / (norm + 1e-6)) * self.dora_m
            
            return F.linear(input_tensor, weight.to(input_tensor.dtype), self.original_layer.bias)
        
        result = self.original_layer(input_tensor)
        
        if self.r > 0:
            # input_tensor shape: (batch, ..., in)
            # lora_a shape: (r, in) -> input_tensor @ A.T -> (batch, ..., r)
            # lora_b shape: (out, r) -> result @ B.T -> (batch, ..., out)
            x_dropped = self.lora_dropout(input_tensor)
            lora_out = (x_dropped @ self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)) * self.scaling
            if self.gate:
                lora_out *= self.lora_gate
            result += lora_out
            
        return result

    def __repr__(self) -> str:
        '''
        Returns a string representation of the module.

        Returns:
            str: String representation.
        '''
        prefix = 'Gated' if self.gate else ''
        suffix = 'DoRA' if self.dora else 'LoRA'
        return f'{self.__class__.__name__}(type={prefix}{suffix}, in_features={self.in_features}, out_features={self.out_features}, r={self.r}, merged={self.merged})'


class Conv2dLoRA(BasicLoRA):
    '''
    Implements Low-Rank Adaptation (LoRA) for Conv2d layers.

    Uses two consecutive convolution layers to simulate low-rank matrix decomposition:
    1. A layer: Reduces channels to r, maintains kernel_size.
    2. B layer: Restores channels, uses 1x1 kernel.
    '''

    def __init__(
        self,
        original_layer: nn.Conv2d,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
        gate: bool = False,
        dora: bool = False,
        gradient_checkpointing: bool = False
    ) -> None:
        '''
        Initializes the Conv2dLoRA module.

        Args:
            original_layer (nn.Conv2d): The original Conv2d layer.
            r (int): The rank of the low-rank adaptation. Defaults to 8.
            lora_alpha (int): The scaling factor for LoRA. Defaults to 16.
            lora_dropout (float): Dropout probability. Defaults to 0.05.
            merge_weights (bool): Whether to merge LoRA weights upon initialization. Defaults to False.
            gate (bool): Whether to use Gated LoRA. Defaults to False.
            dora (bool): Whether to use DoRA. Defaults to False.
            gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
        '''
        super().__init__(
            original_layer, r, lora_alpha, lora_dropout,
            merge_weights, gate, dora, gradient_checkpointing
        )
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups

        if r > 0:
            self.lora_gate = nn.Parameter(torch.tensor([1.0])) if gate else None
            self.lora_a = nn.Conv2d(
                self.in_channels, r,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=False
            )

            self.lora_b = nn.Conv2d(
                r, self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )

            self.lora_dropout = nn.Dropout(p=lora_dropout)

        self.reset_parameters()

        if dora and r > 0:
            # Conv2d weight: (out, in, k, k) -> norm dim=(1,2,3) for each output channel
            self.dora_m = nn.Parameter(
                self.original_layer.weight.norm(p=2, dim=(1, 2, 3), keepdim=True)
            )

        if hasattr(self.original_layer, 'weight'):
            self.to(self.original_layer.weight.device)

        if merge_weights:
            self.merge()

    def reset_parameters(self) -> None:
        '''
        Resets LoRA parameters.
        
        A convolution layer is initialized using Kaiming Uniform, and B is initialized to zero.
        '''
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)

    def merge(self) -> None:
        '''
        Merges LoRA weights into the original convolution layer weights.
        
        Uses einsum to calculate the equivalent convolution kernel and adds it to the original weight.
        '''
        if self.r > 0 and not self.merged:
            if self.dora and self.weight_backup is None:
                self.weight_backup = self.original_layer.weight.data.clone()

            weight_b = self.lora_b.weight.squeeze(3).squeeze(2) # (out, r)
            weight_a = self.lora_a.weight # (r, in, k, k)
            
            # i: out_channels, j: r, k: in_channels, m, n: kernel dims
            delta_w = torch.einsum('ij, jkmn -> ikmn', weight_b, weight_a) * self.scaling
            if self.gate:
                delta_w *= self.lora_gate
            
            if self.dora:
                weight = self.original_layer.weight + delta_w
                norm = weight.norm(p=2, dim=(1, 2, 3), keepdim=True)
                weight = (weight / (norm + 1e-6)) * self.dora_m
                self.original_layer.weight.data = weight.to(self.original_layer.weight.dtype)
            else:
                self.original_layer.weight.data += delta_w
            
            self.merged = True

    def unmerge(self) -> None:
        '''
        Subtracts LoRA weights from the original weights.
        '''
        if self.r > 0 and self.merged:
            if self.dora:
                if self.weight_backup is not None:
                    self.original_layer.weight.data.copy_(self.weight_backup)
                    self.weight_backup = None
                else:
                    print('Warning: DoRA weights cannot be unmerged as no backup was found.')
            else:
                weight_b = self.lora_b.weight.squeeze(3).squeeze(2)
                weight_a = self.lora_a.weight
                delta_w = torch.einsum('ij, jkmn -> ikmn', weight_b, weight_a) * self.scaling
                if self.gate:
                    delta_w *= self.lora_gate
                self.original_layer.weight.data -= delta_w
            
            self.merged = False

    def _forward_impl(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Internal implementation of the forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        if self.r > 0 and self.merged:
            return self.original_layer(input_tensor)
            
        if self.dora and self.r > 0:
            weight_b = self.lora_b.weight.squeeze(3).squeeze(2) # (out, r)
            weight_a = self.lora_a.weight # (r, in, k, k)
            delta_w = torch.einsum('ij, jkmn -> ikmn', weight_b, weight_a) * self.scaling
            if self.gate:
                delta_w *= self.lora_gate
            
            weight = self.original_layer.weight + delta_w
            norm = weight.norm(p=2, dim=(1, 2, 3), keepdim=True)
            weight = (weight / (norm + 1e-6)) * self.dora_m
            
            return F.conv2d(
                input_tensor, weight.to(input_tensor.dtype), self.original_layer.bias, 
                self.stride, self.padding, self.dilation, self.groups
            )
            
        result = self.original_layer(input_tensor)
        
        if self.r > 0:
            x_dropped = self.lora_dropout(input_tensor)
            # input_tensor -> Conv(in, r)[spatial] -> Conv(r, out)[1x1]
            lora_out = self.lora_b(self.lora_a(x_dropped)) * self.scaling
            if self.gate:
                lora_out *= self.lora_gate
            result += lora_out
            
        return result

    def __repr__(self) -> str:
        '''
        Returns a string representation of the module.

        Returns:
            str: String representation.
        '''
        prefix = 'Gated' if self.gate else ''
        suffix = 'DoRA' if self.dora else 'LoRA'
        return f'{self.__class__.__name__}(type={prefix}{suffix}, in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, r={self.r}, merged={self.merged})'


class Conv1dLoRA(BasicLoRA):
    '''
    Implements Low-Rank Adaptation (LoRA) for Conv1d layers.

    Uses two consecutive convolution layers to simulate low-rank matrix decomposition:
    1. A layer: Reduces channels to r, maintains kernel_size.
    2. B layer: Restores channels, uses 1x1 kernel.
    '''

    def __init__(
        self,
        original_layer: nn.Conv1d,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
        gate: bool = False,
        dora: bool = False,
        gradient_checkpointing: bool = False
    ) -> None:
        '''
        Initializes the Conv1dLoRA module.

        Args:
            original_layer (nn.Conv1d): The original Conv1d layer.
            r (int): The rank of the low-rank adaptation. Defaults to 8.
            lora_alpha (int): The scaling factor for LoRA. Defaults to 16.
            lora_dropout (float): Dropout probability. Defaults to 0.05.
            merge_weights (bool): Whether to merge LoRA weights upon initialization. Defaults to False.
            gate (bool): Whether to use Gated LoRA. Defaults to False.
            dora (bool): Whether to use DoRA. Defaults to False.
            gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
        '''
        super().__init__(
            original_layer, r, lora_alpha, lora_dropout,
            merge_weights, gate, dora, gradient_checkpointing
        )
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size[0] # Conv1d kernel_size is a tuple
        self.stride = original_layer.stride[0]
        self.padding = original_layer.padding[0]
        self.dilation = original_layer.dilation[0]
        self.groups = original_layer.groups

        if r > 0:
            self.lora_gate = nn.Parameter(torch.tensor([1.0])) if gate else None
            # A: Dimensional reduction + spatial convolution
            self.lora_a = nn.Conv1d(
                self.in_channels, r,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=False
            )
            # B: Dimensional expansion + point convolution (kernel=1)
            self.lora_b = nn.Conv1d(
                r, self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.lora_dropout = nn.Dropout(p=lora_dropout)

        self.reset_parameters()

        if dora and r > 0:
            # Conv1d weight: (out, in, k) -> norm dim=(1,2)
            self.dora_m = nn.Parameter(
                self.original_layer.weight.norm(p=2, dim=(1, 2), keepdim=True)
            )

        if hasattr(self.original_layer, 'weight'):
            self.to(self.original_layer.weight.device)

        if merge_weights:
            self.merge()

    def reset_parameters(self) -> None:
        '''
        Resets LoRA parameters.
        
        A convolution layer is initialized using Kaiming Uniform, and B is initialized to zero.
        '''
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)

    def merge(self) -> None:
        '''
        Merges LoRA weights into the original convolution layer weights.
        '''
        if self.r > 0 and not self.merged:
            if self.dora and self.weight_backup is None:
                self.weight_backup = self.original_layer.weight.data.clone()

            # B: (out, r, 1) -> (out, r)
            weight_b = self.lora_b.weight.squeeze(2)
            # A: (r, in, k)
            weight_a = self.lora_a.weight
            
            # einsum: ij(out,r), jkn(r,in,k) -> ikn(out,in,k)
            delta_w = torch.einsum('ij, jkn -> ikn', weight_b, weight_a) * self.scaling
            if self.gate:
                delta_w *= self.lora_gate
            
            if self.dora:
                weight = self.original_layer.weight + delta_w
                norm = weight.norm(p=2, dim=(1, 2), keepdim=True)
                weight = (weight / (norm + 1e-6)) * self.dora_m
                self.original_layer.weight.data = weight.to(self.original_layer.weight.dtype)
            else:
                self.original_layer.weight.data += delta_w
            
            self.merged = True

    def unmerge(self) -> None:
        '''
        Subtracts LoRA weights from the original weights.
        '''
        if self.r > 0 and self.merged:
            if self.dora:
                if self.weight_backup is not None:
                    self.original_layer.weight.data.copy_(self.weight_backup)
                    self.weight_backup = None
                else:
                    print('Warning: DoRA weights cannot be unmerged as no backup was found.')
            else:
                weight_b = self.lora_b.weight.squeeze(2)
                weight_a = self.lora_a.weight
                delta_w = torch.einsum('ij, jkn -> ikn', weight_b, weight_a) * self.scaling
                if self.gate:
                    delta_w *= self.lora_gate
                self.original_layer.weight.data -= delta_w
            
            self.merged = False

    def _forward_impl(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Internal implementation of the forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        if self.r > 0 and self.merged:
            return self.original_layer(input_tensor)
            
        if self.dora and self.r > 0:
            weight_b = self.lora_b.weight.squeeze(2)
            weight_a = self.lora_a.weight
            delta_w = torch.einsum('ij, jkn -> ikn', weight_b, weight_a) * self.scaling
            if self.gate:
                delta_w *= self.lora_gate
            
            weight = self.original_layer.weight + delta_w
            norm = weight.norm(p=2, dim=(1, 2), keepdim=True)
            weight = (weight / (norm + 1e-6)) * self.dora_m
            
            return F.conv1d(
                input_tensor, weight.to(input_tensor.dtype), self.original_layer.bias,
                self.stride, self.padding, self.dilation, self.groups
            )
            
        result = self.original_layer(input_tensor)
        if self.r > 0:
            x_dropped = self.lora_dropout(input_tensor)
            lora_out = self.lora_b(self.lora_a(x_dropped)) * self.scaling
            if self.gate:
                lora_out *= self.lora_gate
            result += lora_out
        return result

    def __repr__(self) -> str:
        '''
        Returns a string representation of the module.

        Returns:
            str: String representation.
        '''
        prefix = 'Gated' if self.gate else ''
        suffix = 'DoRA' if self.dora else 'LoRA'
        return f'{self.__class__.__name__}(type={prefix}{suffix}, in={self.in_channels}, out={self.out_channels}, kernel={self.kernel_size}, r={self.r}, merged={self.merged})'


class EmbeddingLoRA(BasicLoRA):
    '''
    Implements Low-Rank Adaptation (LoRA) for Embedding layers.

    Adapts Embedding weights by injecting low-rank matrices.
    Formula: h = W_0[idx] + (A[idx] @ B.T) * scaling
    '''

    def __init__(
        self,
        original_layer: nn.Embedding,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
        gate: bool = False,
        dora: bool = False,
        gradient_checkpointing: bool = False
    ) -> None:
        '''
        Initializes the EmbeddingLoRA module.

        Args:
            original_layer (nn.Embedding): The original Embedding layer.
            r (int): The rank of the low-rank adaptation. Defaults to 8.
            lora_alpha (int): The scaling factor for LoRA. Defaults to 16.
            lora_dropout (float): Dropout probability for LoRA path. Defaults to 0.05.
            merge_weights (bool): Whether to merge LoRA weights upon initialization. Defaults to False.
            gate (bool): Whether to use Gated LoRA. Defaults to False.
            dora (bool): Whether to use DoRA. Defaults to False.
            gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
        '''
        super().__init__(
            original_layer, r, lora_alpha, lora_dropout,
            merge_weights, gate, dora, gradient_checkpointing
        )
        self.num_embeddings = original_layer.num_embeddings
        self.embedding_dim = original_layer.embedding_dim
        self.padding_idx = original_layer.padding_idx

        if r > 0:
            self.lora_gate = nn.Parameter(torch.tensor([1.0])) if gate else None
            # lora_a: (num_embeddings, r)
            self.lora_a = nn.Embedding(
                self.num_embeddings, r,
                padding_idx=self.padding_idx
            )
            # lora_b: (r, embedding_dim)
            self.lora_b = nn.Linear(r, self.embedding_dim, bias=False)
            self.lora_dropout = nn.Dropout(p=lora_dropout)

        self.reset_parameters()

        if dora and r > 0:
            # Embedding weight: (V, D) -> norm dim=1
            self.dora_m = nn.Parameter(
                self.original_layer.weight.norm(p=2, dim=1, keepdim=True)
            )

        if hasattr(self.original_layer, 'weight'):
            self.to(self.original_layer.weight.device)

        if merge_weights:
            self.merge()

    def reset_parameters(self) -> None:
        '''
        Resets LoRA parameters.
        
        Matrix A is initialized with normal distribution, and matrix B is initialized to zero.
        '''
        if self.r > 0:
            nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_b.weight)

    def merge(self) -> None:
        '''
        Merges LoRA weights into the original Embedding layer weights.
        '''
        if self.r > 0 and not self.merged:
            if self.dora and self.weight_backup is None:
                self.weight_backup = self.original_layer.weight.data.clone()

            weight_b = self.lora_b.weight # (D, r)
            weight_a = self.lora_a.weight # (V, r)
            
            delta_w = (weight_a @ weight_b.T) * self.scaling
            if self.gate:
                delta_w *= self.lora_gate
            
            if self.dora:
                weight = self.original_layer.weight + delta_w
                norm = weight.norm(p=2, dim=1, keepdim=True)
                weight = (weight / (norm + 1e-6)) * self.dora_m
                self.original_layer.weight.data = weight.to(self.original_layer.weight.dtype)
            else:
                self.original_layer.weight.data += delta_w
            
            self.merged = True

    def unmerge(self) -> None:
        '''
        Subtracts LoRA weights from the original weights.
        '''
        if self.r > 0 and self.merged:
            if self.dora:
                if self.weight_backup is not None:
                    self.original_layer.weight.data.copy_(self.weight_backup)
                    self.weight_backup = None
                else:
                    print('Warning: DoRA weights cannot be unmerged as no backup was found.')
            else:
                weight_b = self.lora_b.weight
                weight_a = self.lora_a.weight
                delta_w = (weight_a @ weight_b.T) * self.scaling
                if self.gate:
                    delta_w *= self.lora_gate
                self.original_layer.weight.data -= delta_w
            
            self.merged = False

    def _forward_impl(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Internal implementation of the forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor (indices).

        Returns:
            torch.Tensor: Output tensor.
        '''
        if self.r > 0 and self.merged:
            return self.original_layer(input_tensor)
            
        if self.dora and self.r > 0:
            # DoRA embedding
            weight_b = self.lora_b.weight 
            weight_a = self.lora_a.weight 
            delta_w = (weight_a @ weight_b.T) * self.scaling
            if self.gate:
                delta_w *= self.lora_gate
            
            weight = self.original_layer.weight + delta_w
            norm = weight.norm(p=2, dim=1, keepdim=True)
            weight = (weight / (norm + 1e-6)) * self.dora_m
            
            return F.embedding(
                input_tensor, weight.to(input_tensor.dtype if input_tensor.dtype.is_floating_point else self.original_layer.weight.dtype), self.padding_idx, 
                self.original_layer.max_norm, self.original_layer.norm_type,
                self.original_layer.scale_grad_by_freq, self.original_layer.sparse
            )

        result = self.original_layer(input_tensor)
        
        if self.r > 0:
            # A(input_tensor): Look up -> (Batch, Len, r)
            a_out = self.lora_a(input_tensor)
            a_dropped = self.lora_dropout(a_out)
            # B(A(input_tensor)): Linear -> (Batch, Len, Dim)
            lora_out = self.lora_b(a_dropped) * self.scaling
            if self.gate:
                lora_out *= self.lora_gate
            result += lora_out
            
        return result

    def __repr__(self) -> str:
        '''
        Returns a string representation of the module.

        Returns:
            str: String representation.
        '''
        prefix = 'Gated' if self.gate else ''
        suffix = 'DoRA' if self.dora else 'LoRA'
        return f'{self.__class__.__name__}(type={prefix}{suffix}, num={self.num_embeddings}, dim={self.embedding_dim}, r={self.r}, merged={self.merged})'
