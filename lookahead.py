import math
import torch
import torch.nn as nn
import torch.optim as optim

from cortex_hierarchy import CorticalHierarchy


# ─────────────────────────── 1. 数据集: 多步序列规则 ───────────────────────────

class MultiStepSequenceDataset:
    """5 token 循环序列, 由 PFC rule 决定方向 (打破 Z/4Z 对称)"""

    def __init__(self, num_features: int, batch_size: int, device: str = 'cpu'):
        self.F = num_features
        self.B = batch_size
        self.device = device

        # 【修改】5 个 token, 相位 2π/5 等距
        self.token_names = ['A', 'B', 'C', 'D', 'E']
        n_tok = len(self.token_names)
        self.token_phases = [2 * math.pi * i / n_tok for i in range(n_tok)]

        self.tokens = {}
        for name, phase in zip(self.token_names, self.token_phases):
            self.tokens[name] = torch.polar(
                torch.ones(num_features, device=device),
                torch.full((num_features,), phase, device=device)
            )

        # 【修改】5 元循环的两个方向
        self.seq_forward  = ['A', 'B', 'C', 'D', 'E']  # +2π/5
        self.seq_backward = ['A', 'E', 'D', 'C', 'B']  # -2π/5

    def get_token(self, name: str) -> torch.Tensor:
        return self.tokens[name].unsqueeze(0).repeat(self.B, 1)

    def generate(self, current: str, rule: int) -> dict:
        seq = self.seq_forward if rule == 0 else self.seq_backward
        idx = seq.index(current)
        # 【修改】mod 5
        target_names = [seq[(idx + k) % 5] for k in (1, 2, 3)]

        rule_phase = 0.0 if rule == 0 else math.pi
        x_top = torch.polar(
            torch.ones(self.B, self.F, device=self.device) * 1.5,
            torch.full((self.B, self.F), rule_phase, device=self.device)
        )

        return {
            'current':      current,
            'rule':         rule,
            'x_sensory':    self.get_token(current),
            'x_top':        x_top,
            'targets':      [self.get_token(n) for n in target_names],
            'target_names': target_names
        }


# ─────────────────────────── 2. 分类评估 ───────────────────────────

def classify(z: torch.Tensor, tokens: dict) -> list[str]:
    """对每个 batch sample, 在 4 个 token 中找复数余弦最高的"""
    out = []
    for b in range(z.shape[0]):
        best_sim, best_name = -1e9, '?'
        for name, vec in tokens.items():
            sim = (torch.real(torch.sum(z[b] * vec.conj())) /
                   (torch.norm(z[b]) * torch.norm(vec) + 1e-8)).item()
            if sim > best_sim:
                best_sim, best_name = sim, name
        out.append(best_name)
    return out


# ─────────────────────────── 3. 训练 ───────────────────────────

def train(epochs: int = 200, num_passes: int = 2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    F, B, N = 64, 16, 3

    # 用 'apical' gate_phase 因为我们要做相位算术 (旋转 ±π/2)
    # 用 'pre' work_type 因为反馈纠错对多步预测有帮助
    h = CorticalHierarchy(
        num_features = F,
        num_layers   = N,
        work_type    = 'pre',
        gate_phase   = 'apical',
        share_weights = False,
        track        = True
    ).to(device)

    optimizer = optim.Adam(h.parameters(), lr=0.005, weight_decay=1e-5)
    dataset   = MultiStepSequenceDataset(F, B, device)

    # 8 种 (token, rule) 组合
    combos = [(t, r) for t in dataset.token_names for r in (0, 1)]

    print(f"\n分层多步前瞻预测  | N={N} F={F} passes={num_passes} | device={device}")
    print("-" * 78)
    print(f"{'Epoch':<7}{'Loss':<10}{'Step1':<10}{'Step2':<10}{'Step3':<10}{'L0.coinc':<10}{'L2.coinc':<10}")
    print("-" * 78)

    for epoch in range(1, epochs + 1):
        h.train()
        epoch_loss = 0.0

        for current, rule in combos:
            data = dataset.generate(current, rule)
            optimizer.zero_grad()

            _, z_l5s = h(
                x_sensory     = data['x_sensory'],
                x_top_context = data['x_top'],
                num_passes    = num_passes,
                return_all    = True
            )

            # 每层一个目标; 越靠后步数越长, 加权稍降以稳定训练
            weights = [1.0, 0.8, 0.6]
            loss = sum(w * torch.mean(torch.abs(z_l5s[i] - data['targets'][i])**2)
                       for i, w in enumerate(weights))

            loss.backward()
            nn.utils.clip_grad_norm_(h.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # 每 10 epoch 评估
        if epoch == 1 or epoch % 10 == 0:
            h.eval()
            correct = [0, 0, 0]
            total = 0
            l0_coinc, l2_coinc = 0.0, 0.0

            with torch.no_grad():
                for current, rule in combos:
                    data = dataset.generate(current, rule)
                    _, z_l5s = h(data['x_sensory'], data['x_top'],
                                 num_passes=num_passes, return_all=True)

                    for i in range(N):
                        preds = classify(z_l5s[i], dataset.tokens)
                        correct[i] += sum(1 for p in preds if p == data['target_names'][i])
                    total += B

                    l0_coinc += h._col(0).last_coincidence
                    l2_coinc += h._col(N - 1).last_coincidence

            accs = [c / total for c in correct]
            print(f"{epoch:<7}{epoch_loss/len(combos):<10.4f}"
                  f"{accs[0]*100:<10.1f}{accs[1]*100:<10.1f}{accs[2]*100:<10.1f}"
                  f"{l0_coinc/len(combos):<10.4f}{l2_coinc/len(combos):<10.4f}")

            if all(a == 1.0 for a in accs):
                print(f"\n[完美收敛] Epoch {epoch}: 三步预测全部 100%")
                break

    # ─── 决策矩阵 ───
    h.eval()
    print("\n最终决策矩阵 (Step1/Step2/Step3 [Expected]):")
    with torch.no_grad():
        for rule in (0, 1):
            label = 'forward (+π/2)' if rule == 0 else 'backward (-π/2)'
            print(f"  Rule {rule}  {label}:")
            for name in dataset.token_names:
                data = dataset.generate(name, rule)
                _, z_l5s = h(data['x_sensory'], data['x_top'],
                             num_passes=num_passes, return_all=True)
                preds = [classify(z_l5s[i], dataset.tokens)[0] for i in range(N)]
                mark  = '✓' if preds == data['target_names'] else '✗'
                print(f"    {name} → {'/'.join(preds)}  [{'/'.join(data['target_names'])}]  {mark}")


if __name__ == '__main__':
    torch.manual_seed(42)
    train(epochs=200, num_passes=3)