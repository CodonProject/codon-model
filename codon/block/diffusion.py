from codon import *
from codon.block.mlp import MLP


class MMDiTBlock(BasicModel):
    def __init__(self, dim, cond_dim, mlp_ratio=4.0, use_swiglu: bool = False, bidirectional_cross_attn=False):
        super().__init__()
        self.dim = dim
        self.bidirectional_cross_attn = bidirectional_cross_attn
        self.use_swiglu = use_swiglu
        
        self.norm1_x = nn.RMSNorm(dim)
        self.norm1_c = nn.RMSNorm(dim)
        
        self.to_q_x = nn.Linear(dim, dim)
        self.to_k_x = nn.Linear(dim, dim)
        self.to_v_x = nn.Linear(dim, dim)
        
        self.to_q_c = nn.Linear(dim, dim)
        self.to_k_c = nn.Linear(dim, dim)
        self.to_v_c = nn.Linear(dim, dim)
        
        self.to_out_x = nn.Linear(dim, dim)
        self.to_out_c = nn.Linear(dim, dim) if bidirectional_cross_attn else None
        
        self.norm2_x = nn.RMSNorm(dim)
        self.norm2_c = nn.RMSNorm(dim)
        
        hidden_dim = int(dim * mlp_ratio)
        self.mlp_x = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        ) if not use_swiglu else MLP.SwiGLU(dim, bias=True)
        self.mlp_c = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        ) if not use_swiglu else MLP.SwiGLU(dim, bias=True)
        
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 10 * dim)
        )

    def forward(self, x, c, y):
        '''
        x: 图像特征 [Batch, Seq_img, Dim]
        c: 文本特征 [Batch, Seq_txt, Dim]
        y: 全局条件向量 (时间步+池化文本) [Batch, Dim]
        '''
        (   alpha_x, beta_x, alpha_c, beta_c,
            delta_x, epsilon_x, delta_c, epsilon_c,
            gamma_x, gamma_c
        ) = self.modulation(y).chunk(10, dim=-1)

        def expand_mod(t): return t.unsqueeze(1)

        x_norm = self.norm1_x(x)
        c_norm = self.norm1_c(c)
        
        x_mod = x_norm * expand_mod(alpha_x) + expand_mod(beta_x)
        c_mod = c_norm * expand_mod(alpha_c) + expand_mod(beta_c)
        
        q_x, k_x, v_x = self.to_q_x(x_mod), self.to_k_x(x_mod), self.to_v_x(x_mod)
        q_c, k_c, v_c = self.to_q_c(c_mod), self.to_k_c(c_mod), self.to_v_c(c_mod)
        
        if self.bidirectional_cross_attn:
            attn_x = F.scaled_dot_product_attention(q_x, k_c, v_c, is_causal=False)
            attn_c = F.scaled_dot_product_attention(q_c, k_x, v_x, is_causal=False)
            
            x = x + self.to_out_x(attn_x) * expand_mod(gamma_x)
            c = c + self.to_out_c(attn_c) * expand_mod(gamma_c)
        else:
            attn_x = F.scaled_dot_product_attention(q_x, k_c, v_c, is_causal=False)
            
            x = x + self.to_out_x(attn_x) * expand_mod(gamma_x)

        x_norm = self.norm2_x(x)
        c_norm = self.norm2_c(c)
        
        x_mod_mlp = x_norm * expand_mod(delta_x) + expand_mod(epsilon_x)
        c_mod_mlp = c_norm * expand_mod(delta_c) + expand_mod(epsilon_c)
        
        x_mlp_out = self.mlp_x(x_mod_mlp)
        c_mlp_out = self.mlp_c(c_mod_mlp)
        
        x = x + x_mlp_out * expand_mod(gamma_x)
        c = c + c_mlp_out * expand_mod(gamma_c)

        return x, c


class MMDiTList(BasicModel):
    def __init__(
        self,
        model_dim: int,
        cond_dim: int,
        num_block: int,
        mlp_ratio: float = 4.0,
        use_swiglu: bool = False,
        bidirectional_cross_attn=False,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.cond_dim = cond_dim
        self.num_block = num_block
        self.mlp_ratio = mlp_ratio
        self.use_swiglu = use_swiglu
        self.bidirectional_cross_attn = bidirectional_cross_attn

        self.features = nn.ModuleList([
            MMDiTBlock(
                model_dim,
                cond_dim,
                mlp_ratio=mlp_ratio,
                use_swiglu=use_swiglu,
                bidirectional_cross_attn=bidirectional_cross_attn
            ) for _ in range(num_block)
        ])

    def __len__(self):
        return self.num_block
    
    def forward(self, x, c, y):
        '''
        x: 图像特征 [Batch, Seq_img, Dim]
        c: 文本特征 [Batch, Seq_txt, Dim]
        y: 全局条件向量 [Batch, Dim]
        '''
        for block in self.features: x, c = block(x, c, y)
        
        return x, c
