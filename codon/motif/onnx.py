from codon.base import *
from codon.utils.onnx import patch_rms_norm

from typing import TYPE_CHECKING, Tuple, Optional, Any

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

    def _build_export_args(
        self,
        float_dtype: torch.dtype = torch.float32
    ) -> Tuple[Tuple[Any, ...], list, list, dict]:
        '''
        Constructs dummy inputs, input/output names and dynamic axes spec for ONNX export.

        Args:
            float_dtype (torch.dtype): The dtype to use for floating-point dummy
                tensors (mask and past key/value buffers). Integer inputs
                (input_ids, start_pos) are unaffected. Defaults to torch.float32.

        Returns:
            Tuple[Tuple[Any, ...], list, list, dict]: A tuple of
                (dummy_inputs, input_names, output_names, dynamic_axes).
        '''
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
        # 4D empty tensor representing "no mask" while preserving rank for downstream ops
        dummy_mask = torch.empty(1, 1, 0, 0, dtype=float_dtype)

        dummy_past_key_values = []
        for _ in range(num_layers):
            dummy_past_key_values.append(
                torch.zeros(batch_size, num_kv_heads, past_seq_len, head_dim, dtype=float_dtype)
            )
            dummy_past_key_values.append(
                torch.zeros(batch_size, num_kv_heads, past_seq_len, head_dim, dtype=float_dtype)
            )

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

        dummy_inputs = (dummy_input_ids, dummy_start_pos, dummy_mask, *dummy_past_key_values)
        return dummy_inputs, input_names, output_names, dynamic_axes

    def _export_raw(
        self,
        target_path: str,
        opset_version: int,
        float_dtype: torch.dtype = torch.float32
    ) -> None:
        '''
        Performs the ONNX export via `torch.onnx.export`.

        Args:
            target_path (str): The path to write the ONNX model to.
            opset_version (int): The ONNX opset version.
            float_dtype (torch.dtype): Floating-point dtype for dummy inputs.
                Should match the model's parameter dtype. Defaults to float32.
        '''
        dummy_inputs, input_names, output_names, dynamic_axes = self._build_export_args(
            float_dtype=float_dtype
        )
        torch.onnx.export(
            self,
            dummy_inputs,
            target_path,
            export_params=True,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

    def _simplify_model(self, model_proto: Any) -> Any:
        '''
        Applies static graph optimization via `onnxsim`.

        Constant folding, redundant Cast/Reshape elimination, and shape inference
        based simplifications are performed on the graph. Falls back to the
        original model with a warning if validation fails.

        Args:
            model_proto: The loaded ONNX model proto.

        Returns:
            The simplified ONNX model proto, or the original on failure.
        '''
        try:
            import onnxsim
        except ImportError as exc:
            raise ImportError(
                'simplify=True requires onnxsim. Install via: pip install onnxsim'
            ) from exc

        import warnings
        try:
            simplified, check = onnxsim.simplify(model_proto)
        except Exception as exc:
            warnings.warn(f'onnxsim raised {type(exc).__name__}: {exc}; using unsimplified model')
            return model_proto

        if not check:
            warnings.warn('onnxsim validation failed; using unsimplified model')
            return model_proto
        return simplified

    def _save_model(
        self,
        model_proto: Any,
        onnx_path: str,
        external_data_path: Optional[str]
    ) -> None:
        '''
        Saves the ONNX model proto, optionally externalizing weights.

        Args:
            model_proto: The ONNX model proto to save.
            onnx_path (str): The target file path for the ONNX model.
            external_data_path (Optional[str]): Relative location of the external
                weights file. If None, weights are embedded.
        '''
        import onnx
        if external_data_path is not None:
            onnx.save_model(
                model_proto,
                onnx_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data_path,
                size_threshold=1024
            )
        else:
            onnx.save(model_proto, onnx_path)

    def export(
        self,
        onnx_path: str,
        external_data_path: Optional[str] = None,
        opset_version: int = 14,
        precision: str = 'fp32',
        simplify: bool = False
    ) -> None:
        '''
        Exports the wrapped causal language model to an ONNX computation graph.

        The export pipeline runs in this order:
            1. (FP16 only) Convert model parameters to FP16 via `self.half()`.
            2. ONNX export via `torch.onnx.export` with dummy inputs whose
               floating-point tensors match the model dtype.
            3. Optional static graph simplification via `onnxsim`.
            4. Final save (optionally with external data).

        Args:
            onnx_path (str): The target file path to save the ONNX model.
            external_data_path (Optional[str]): Path to the external binary file
                for storing model weights. If None, weights are embedded.
                Defaults to None.
            opset_version (int): The ONNX opset version. Defaults to 14.
            precision (str): Numerical precision of the exported model.
                Either 'fp32' (default) or 'fp16'. FP16 roughly halves the
                stored model size at the cost of some numerical precision;
                RMSNorm internally upcasts to FP32 to preserve stability
                (see `codon.utils.onnx.rms_norm_onnx_forward`).
                When `precision='fp16'`, the runtime inputs `mask` and all
                `past_key_*` / `past_value_*` tensors must be passed as FP16;
                `input_ids` and `start_pos` remain int64.
            simplify (bool): If True, run `onnxsim` static graph simplification
                (constant folding, redundant op elimination) on the exported
                graph. Defaults to False.

        Raises:
            ValueError: If `precision` is not one of {'fp32', 'fp16'}.
            ImportError: If `simplify=True` but `onnxsim` is not installed.
        '''
        import os

        if precision not in ('fp32', 'fp16'):
            raise ValueError(
                f"precision must be 'fp32' or 'fp16', got {precision!r}"
            )

        self.eval()

        # Step 1: precision conversion (must happen before tracing so the
        # exported graph is type-consistent throughout)
        if precision == 'fp16':
            self.half()
            float_dtype = torch.float16
        else:
            float_dtype = torch.float32

        needs_post_processing = simplify
        needs_external = external_data_path is not None

        # Step 2: ONNX export
        # Use a temp path whenever we will rewrite the file afterwards
        if needs_post_processing or needs_external:
            temp_path = onnx_path + '.temp'
            self._export_raw(temp_path, opset_version, float_dtype=float_dtype)
        else:
            temp_path = None
            self._export_raw(onnx_path, opset_version, float_dtype=float_dtype)

        # Steps 3-4: optional simplify and final save
        if needs_post_processing or needs_external:
            import onnx
            model_proto = onnx.load(temp_path)

            if simplify:
                model_proto = self._simplify_model(model_proto)

            self._save_model(model_proto, onnx_path, external_data_path)

            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)
