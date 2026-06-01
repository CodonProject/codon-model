from codon.base import *
from codon.utils.onnx import patch_rms_norm

from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from codon.motif.base import CausalLanguageModel, CausalLanguageModelOutput


class CausalLanguageModelONNXWrapper(BasicModel):
    '''
    ONNX-compatible wrapper for causal language models.

    This wrapper flattens the nested input and output structures (specifically
    the past key values) of a CausalLanguageModel to satisfy ONNX export
    constraints, enabling seamless integration with ONNX Runtime.

    Attributes:
        model (CausalLanguageModel): The wrapped causal language model.
        num_layers (int): The number of decoder layers.
    '''

    def __init__(self, model: 'CausalLanguageModel') -> None:
        '''
        Initializes the CausalLanguageModelONNXWrapper.

        Args:
            model (CausalLanguageModel): The CausalLanguageModel instance to wrap.
        '''
        super().__init__()
        self.model = model
        self.num_layers = len(model.decoder)
        patch_rms_norm(self.model)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *past_key_values: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        '''
        Forward pass for the causal language model ONNX wrapper.

        Args:
            input_ids (torch.Tensor): Input token IDs. Shape: (batch_size, seq_len).
            start_pos (torch.Tensor): The starting position index as a tensor.
            mask (Optional[torch.Tensor]): Attention mask. Defaults to None.
                An empty tensor of shape (0,) is treated as None.
            *past_key_values (torch.Tensor): Flat sequence of past key and value
                tensors for each layer. There should be 2 * num_layers tensors:
                (K_0, V_0, K_1, V_1, ...).

        Returns:
            Tuple[torch.Tensor, ...]: A tuple where the first element is the logits
                tensor of shape (batch_size, seq_len, vocab_size), followed by the
                updated flat key and value tensors for each layer:
                (logits, new_K_0, new_V_0, new_K_1, new_V_1, ...).
        '''
        reconstructed_pkv = []
        if len(past_key_values) > 0:
            for i in range(self.num_layers):
                k = past_key_values[2 * i]
                v = past_key_values[2 * i + 1]
                reconstructed_pkv.append((k, v))
        else:
            reconstructed_pkv = None

        # Check mask shape dynamically to avoid trace-time scalar constant conversion warning
        is_empty_mask = False
        if mask is not None:
            mask_shape = mask.shape
            if len(mask_shape) == 1 and mask_shape[0] == 0:
                is_empty_mask = True
            elif len(mask_shape) == 0:
                is_empty_mask = True

        actual_mask = None if is_empty_mask else mask

        outputs: CausalLanguageModelOutput = self.model(
            input_ids=input_ids,
            mask=actual_mask,
            start_pos=start_pos,
            past_key_values=reconstructed_pkv,
            use_cache=True
        )

        flat_outputs = [outputs.logits]
        if outputs.past_key_values is not None:
            for k, v in outputs.past_key_values:
                flat_outputs.append(k)
                flat_outputs.append(v)

        return tuple(flat_outputs)

    def export(
        self,
        onnx_path: str,
        external_data_path: Optional[str] = None,
        opset_version: int = 14
    ) -> None:
        '''
        Exports the wrapped causal language model to an ONNX computation graph.

        Args:
            onnx_path (str): The target file path to save the ONNX model.
            external_data_path (Optional[str]): Path to the external binary file
                for storing model weights. If None, weights are embedded.
                Defaults to None.
            opset_version (int): The ONNX opset version. Defaults to 14.
        '''
        import os
        import torch

        self.eval()

        # Retrieve model architecture parameters dynamically
        vocab_size = getattr(self.model, 'vocab_size', 8192)
        model_dim = getattr(self.model, 'model_dim', 768)
        num_layers = self.num_layers

        first_layer = self.model.decoder[0]
        num_heads = getattr(first_layer, 'num_heads', None)
        if num_heads is None and hasattr(first_layer, 'attn'):
            num_heads = getattr(first_layer.attn, 'num_heads', None)
        if num_heads is None:
            num_heads = 8

        num_kv_heads = getattr(first_layer, 'num_kv_heads', None)
        if num_kv_heads is None and hasattr(first_layer, 'attn'):
            num_kv_heads = getattr(first_layer.attn, 'num_kv_heads', None)
        if num_kv_heads is None:
            num_kv_heads = 2

        head_dim = model_dim // num_heads

        # Prepare dummy inputs for safe tracing
        batch_size = 1
        seq_len = 4
        past_seq_len = 2

        dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        dummy_start_pos = torch.tensor([0], dtype=torch.long)
        # Using a 4D empty tensor for attention mask to prevent warnings or layout trace issues if needed,
        # or a flat 0D tensor with size 0 that dynamically represents "No mask".
        dummy_mask = torch.empty(1, 1, 0, 0, dtype=torch.float32)

        dummy_past_key_values = []
        for _ in range(num_layers):
            dummy_past_key_values.append(torch.zeros(batch_size, num_kv_heads, past_seq_len, head_dim))
            dummy_past_key_values.append(torch.zeros(batch_size, num_kv_heads, past_seq_len, head_dim))

        input_names = ['input_ids', 'start_pos', 'mask']
        for i in range(num_layers):
            input_names.append(f'past_key_{i}')
            input_names.append(f'past_value_{i}')

        output_names = ['logits']
        for i in range(num_layers):
            output_names.append(f'present_key_{i}')
            output_names.append(f'present_value_{i}')

        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'start_pos': {0: 'one_dim'},
            'logits': {0: 'batch_size', 1: 'seq_len'},
            'mask': {0: 'batch_size', 2: 'seq_len', 3: 'total_seq_len'}
        }
        for i in range(num_layers):
            dynamic_axes[f'past_key_{i}'] = {0: 'batch_size', 2: 'past_seq_len'}
            dynamic_axes[f'past_value_{i}'] = {0: 'batch_size', 2: 'past_seq_len'}
            dynamic_axes[f'present_key_{i}'] = {0: 'batch_size', 2: 'total_seq_len'}
            dynamic_axes[f'present_value_{i}'] = {0: 'batch_size', 2: 'total_seq_len'}

        # Handle path routing for external data vs embedded
        export_path = onnx_path
        temp_path = None
        if external_data_path is not None:
            temp_path = onnx_path + '.temp'
            export_path = temp_path

        # Export unified graph
        torch.onnx.export(
            self,
            (dummy_input_ids, dummy_start_pos, dummy_mask, *dummy_past_key_values),
            export_path,
            export_params=True,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

        # Externalize weights if requested
        if external_data_path is not None:
            import onnx
            model_proto = onnx.load(temp_path)
            onnx.save_model(
                model_proto,
                onnx_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data_path,
                size_threshold=1024
            )
            if os.path.exists(temp_path):
                os.remove(temp_path)
