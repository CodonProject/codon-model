from codon import *
from codon.ops.attention_residuals import (
    AttnResState,
    AttnResOutput,
    apply_block_attn_res,
)


class AttnResWrapper(BasicModel):
    def __init__(
        self, 
        module: nn.Module, 
        d_model: int, 
        is_block_boundary: bool = False
    ):
        super().__init__()
        self.module = module
        self.is_block_boundary = is_block_boundary
        
        self.attn_res_norm = nn.RMSNorm(d_model)
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))

    def _extract_main_output(self, module_output: Any) -> Tuple[torch.Tensor, Any]:
        if isinstance(module_output, torch.Tensor):
            return module_output, None
        elif isinstance(module_output, (tuple, list)):
            return module_output[0], module_output[1:]
        elif isinstance(module_output, dict):
            if 'hidden_states' in module_output:
                return module_output['hidden_states'], module_output
            elif 'output' in module_output:
                return module_output['output'], module_output
            return next(iter(module_output.values())), module_output
        elif hasattr(module_output, 'hidden_states'):
            return module_output.hidden_states, module_output
        elif hasattr(module_output, 'output'):
            return module_output.output, module_output
        else:
            raise TypeError(f'Unsupported output type: {type(module_output)}')

    def forward(
        self, 
        state: AttnResState, 
        *args, 
        **kwargs
    ) -> AttnResOutput:
        h = apply_block_attn_res(
            blocks=state.blocks,
            partial_block=state.partial_block,
            weight=self.pseudo_query,
            norm=self.attn_res_norm
        )
        
        raw_output = self.module(h, *args, **kwargs)
        main_output, aux_outputs = self._extract_main_output(raw_output)
        
        if state.partial_block is None:
            new_partial = main_output
        else:
            new_partial = state.partial_block + main_output
            
        new_blocks = list(state.blocks)
        if self.is_block_boundary:
            new_blocks.append(new_partial)
            new_partial = None
        
        new_state = AttnResState(blocks=new_blocks, partial_block=new_partial)
        
        return AttnResOutput(
            hidden_states=h,
            state=new_state,
            aux_outputs=aux_outputs
        )