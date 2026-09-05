from codon import *
from codon.exp.block.cortex_v2 import CorticalColumn
from typing import Optional, Literal


class CorticalHierarchy(BasicModel):
    """多列分层皮层柱网络。

    数据流:
      自下而上:  col[i].z_l23 → col[i+1].x_l4
      自上而下:  col[i+1].z_l5 → col[i].x_pfc  (经多次扫描携带)
    
    Args:
        num_features:    每层特征维度 (所有层共享)
        num_layers:      皮层柱数量
        work_type:       BAC 工作模式 ('pre' / 'ctc')
        gate_phase:      门相位语义 ('apical' / 'real')
        share_weights:   是否在所有层共享同一组权重
        track:           是否暴露中间状态
    """
    def __init__(
        self,
        num_features: int,
        num_layers: int,
        work_type: Literal['ctc', 'pre'] = 'pre',
        gate_phase: Literal['real', 'apical'] = 'apical',
        share_weights: bool = False,
        track: bool = True
    ):
        super().__init__()
        assert num_layers >= 1
        self.num_features  = num_features
        self.num_layers    = num_layers
        self.share_weights = share_weights
        self.track         = track

        col_kwargs = dict(
            num_features=num_features,
            work_type=work_type,
            gate_phase=gate_phase,
            track=track
        )

        if share_weights:
            # 单组权重, N 次复用 (类似 Universal Transformer)
            self._shared = CorticalColumn(**col_kwargs)
            self.columns = None
        else:
            self._shared = None
            self.columns = nn.ModuleList([
                CorticalColumn(**col_kwargs) for _ in range(num_layers)
            ])

        # 中间状态 (forward 后填充)
        self.last_z_l23s = []
        self.last_z_l5s  = []

    def _col(self, i: int) -> CorticalColumn:
        return self._shared if self.share_weights else self.columns[i]
    
    def forward(
        self,
        x_sensory: torch.Tensor,
        x_top_context: Optional[torch.Tensor] = None,
        x_thalamus_phase: Optional[torch.Tensor] = None,
        x_ach: Optional[torch.Tensor] = None,
        num_passes: int = 2,
        return_all: bool = False
    ):
        """
        Returns:
            return_all=False: (z_l23_top, z_l5_top)
            return_all=True : (list[z_l23], list[z_l5])  按从底到顶排列
        """
        N = self.num_layers
        prev_z_l5s = [None] * N
        z_l23s = [None] * N
        z_l5s  = [None] * N

        for _ in range(num_passes):
            x_l4_i = x_sensory
            for i in range(N):
                if i == N - 1:
                    x_pfc_i = x_top_context
                else:
                    x_pfc_i = prev_z_l5s[i + 1]

                z_l23_i, z_l5_i = self._col(i)(
                    x_l4=x_l4_i,
                    x_pfc=x_pfc_i,
                    x_thalamus_phase=x_thalamus_phase,
                    x_ach=x_ach
                )
                z_l23s[i] = z_l23_i
                z_l5s[i]  = z_l5_i
                x_l4_i    = z_l23_i

            prev_z_l5s = [z.clone() if z is not None else None for z in z_l5s]

        if self.track:
            with torch.no_grad():
                self.last_z_l23s = [z.detach() for z in z_l23s]
                self.last_z_l5s  = [z.detach() for z in z_l5s]

        if return_all:
            return z_l23s, z_l5s
        return z_l23s[-1], z_l5s[-1]