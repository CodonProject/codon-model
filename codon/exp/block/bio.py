from codon import *
from codon.utils.data.base import CodonDataset

@dataclass
class EABStats:
    capacity: int
    used_slots: int
    active_ids: List[int]
    canvas_norm: float

@dataclass
class Episode:
    id: int
    key: torch.Tensor
    value: torch.Tensor

@dataclass
class Association:
    topk: int
    ids: List[int]
    scores: torch.Tensor


class EABReplayDataset(CodonDataset):
    def __init__(self, data):
        self._data = data
        
    def __len__(self) -> int:
        return len(self._data)
        
    def __getitem__(self, idx: int):
        return self._data[idx]


class EpisodicAssociativeBlock(BasicModel):
    def __init__(
        self,
        value_dim: int,
        key_dim: int,
        capacity: int,
        sparsity: float = 0.05
    ):
        super().__init__()
        self.value_dim = value_dim
        self.key_dim = key_dim
        self.capacity = capacity
        
        # 钥匙矩阵 K [Capacity, D_key] (冻结)
        keys = torch.randn(capacity, key_dim)
        threshold = torch.quantile(keys, 1.0 - sparsity, dim=1, keepdim=True)
        keys = torch.where(keys >= threshold, keys, torch.zeros_like(keys))

        # L2 归一化保证多次内积重构时的尺度稳定
        keys = keys / (torch.norm(keys, dim=1, keepdim=True) + 1e-8) 
        self.register_buffer('K', keys)
        
        # 记忆矩阵 W [D_val, D_key]
        self.register_buffer('W', torch.zeros(value_dim, key_dim))
        
        # 槽位使用统计器 C [Capacity]
        self.register_buffer('usage_counts', torch.zeros(capacity, dtype=torch.long))

    @property
    def stats(self) -> EABStats:
        active_mask = self.usage_counts > 0
        return EABStats(
            capacity=self.capacity,
            used_slots=int(active_mask.sum().item()),
            active_ids=torch.nonzero(active_mask).squeeze(-1).tolist(),
            canvas_norm=self.W.norm().item()
        )
    
    def memorize(self, value: torch.Tensor, id: Optional[int] = None, gamma: float = 1.0) -> Episode:
        if id is None:
            available_ids = torch.nonzero(self.usage_counts == 0).squeeze(-1)
            if len(available_ids) == 0:
                raise RuntimeError('EAB capacity exhausted.')
            id = int(available_ids[0].item())
            
        key = self.K[id]
        
        # W = W + gamma * (V ⊗ K)
        self.W += gamma * torch.outer(value, key)
        self.usage_counts[id] += 1
        
        return Episode(id=id, key=key, value=value)
    
    def memorize_batch(
        self, 
        values: torch.Tensor,
        ids: torch.Tensor,
        gammas: torch.Tensor
    ):
        '''使用矩阵乘法进行 O(1) 复杂度的批量外积叠加'''
        keys = self.K[ids]      # [B, D_key]
        
        # 广播缩放因子：[B, D_val] * [B, 1]
        scaled_values = values * gammas.unsqueeze(-1)
        
        # 数学魔法：批量外积之和 == 转置矩阵相乘
        # W += V^T @ K
        self.W += torch.matmul(scaled_values.t(), keys)
        
        # 批量更新 usage_counts (使用 scatter_add 避免相同 id 被覆盖)
        ones = torch.ones_like(ids, dtype=self.usage_counts.dtype)
        self.usage_counts.scatter_add_(0, ids, ones)
    
    def recall(self, id: int) -> Episode:
        '''正交解耦读取'''
        if self.usage_counts[id] == 0:
            raise ValueError(f'Slot {id} is empty.')
            
        key = self.K[id]
        value_hat = torch.matmul(self.W, key)
        
        value_hat = value_hat / self.usage_counts[id].float()
        
        return Episode(id=id, key=key, value=value_hat)
    
    def associate(self, value: torch.Tensor, topk: int = 1) -> Association:
        '''联想碰撞，用于前向寻找最相似的记忆槽位'''
        projected = torch.matmul(self.W.t(), value)
        scores = torch.matmul(self.K, projected)
        mask = (self.usage_counts > 0).float()
        scores = scores * mask - (1 - mask) * 1e9 
        
        valid_count = int(mask.sum().item())
        if valid_count == 0:
            return Association(topk=0, ids=[], scores=torch.empty(0))
            
        actual_k = min(topk, valid_count)
        top_scores, top_ids = torch.topk(scores, k=actual_k)
        
        return Association(
            topk=actual_k,
            ids=top_ids.tolist(),
            scores=top_scores
        )
    
    def auto_memorize(
        self,
        class_index: int,
        total_class: int,
        value: torch.Tensor,
        gamma: float = 1.0
    ) -> Episode:
        '''
        自动聚类与记忆路由机制
        自行推断新颖性阈值，并自动调度 consolidate 机制进行记忆碎片整理
        '''
        slots_per_class = self.capacity // total_class
        start_id = class_index * slots_per_class
        end_id = start_id + slots_per_class
        
        class_slot_ids = list(range(start_id, end_id))
        used_ids = [i for i in class_slot_ids if self.usage_counts[i] > 0]
        empty_ids = [i for i in class_slot_ids if self.usage_counts[i] == 0]
        
        target_id = None
        dynamic_gamma = gamma
        
        if len(used_ids) > 0:
            stored_values = torch.stack([self.recall(i).value for i in used_ids]) # [Num_used, D_val]
            
            val_norm = value / (value.norm() + 1e-8)
            stored_norm = stored_values / (stored_values.norm(dim=1, keepdim=True) + 1e-8)
            similarities = torch.matmul(stored_norm, val_norm) # [Num_used]
            
            best_sim, best_idx = torch.max(similarities, dim=0)
            best_id = used_ids[best_idx.item()]
            
            usage_ratio = len(used_ids) / len(class_slot_ids)
            
            if len(used_ids) >= 2:
                sim_matrix = torch.matmul(stored_norm, stored_norm.t())
                mask = ~torch.eye(len(used_ids), dtype=torch.bool, device=self.W.device)
                mean_internal_sim = sim_matrix[mask].mean().item()
                base_thresh = mean_internal_sim + 0.05
            else:
                base_thresh = 0.85
                
            inferred_threshold = base_thresh - (usage_ratio ** 2) * 0.20
            inferred_threshold = max(0.50, min(0.95, inferred_threshold))
            
            if best_sim.item() < inferred_threshold:
                if len(empty_ids) > 0:
                    target_id = empty_ids[0]
                else:
                    if self.consolidate(class_index, total_class, threshold=inferred_threshold):
                        empty_ids = [i for i in class_slot_ids if self.usage_counts[i] == 0]
                        target_id = empty_ids[0]
                    else:
                        target_id = best_id
                        count = self.usage_counts[target_id].item()
                        dynamic_gamma = gamma / (count + 1.0)
            else:
                target_id = best_id
                count = self.usage_counts[target_id].item()
                dynamic_gamma = gamma / (count + 1.0)
        else:
            target_id = empty_ids[0]
            
        return self.memorize(value, id=target_id, gamma=dynamic_gamma)

    def auto_memorize_batch(
        self,
        class_indices: Union[int, torch.Tensor],
        total_class: int,
        values: torch.Tensor,
        base_gamma: float = 1.0
    ):
        '''适应 Batch Size 的自动聚类与记忆路由机制'''
        batch_size = values.size(0)

        if isinstance(class_indices, int):
            class_indices = torch.full((batch_size,), class_indices, dtype=torch.long, device=values.device)
        elif isinstance(class_indices, torch.Tensor) and class_indices.dim() == 0:
            class_indices = class_indices.expand(batch_size)

        target_ids = []
        dynamic_gammas = []
        
        for i in range(batch_size):
            val = values[i]
            c_idx = class_indices[i].item()
            
            slots_per_class = self.capacity // total_class
            start_id = c_idx * slots_per_class
            end_id = start_id + slots_per_class
            
            class_slot_ids = list(range(start_id, end_id))
            used_ids = [idx for idx in class_slot_ids if self.usage_counts[idx] > 0]
            empty_ids = [idx for idx in class_slot_ids if self.usage_counts[idx] == 0]
            
            target_id = None
            gamma_i = base_gamma
            
            if len(used_ids) > 0:
                stored_values = torch.stack([self.recall(idx).value for idx in used_ids])
                val_norm = val / (val.norm() + 1e-8)
                stored_norm = stored_values / (stored_values.norm(dim=1, keepdim=True) + 1e-8)
                similarities = torch.matmul(stored_norm, val_norm)
                
                best_sim, best_idx = torch.max(similarities, dim=0)
                best_id = used_ids[best_idx.item()]
                usage_ratio = len(used_ids) / len(class_slot_ids)
                
                if len(used_ids) >= 2:
                    sim_matrix = torch.matmul(stored_norm, stored_norm.t())
                    mask = ~torch.eye(len(used_ids), dtype=torch.bool, device=self.W.device)
                    mean_internal_sim = sim_matrix[mask].mean().item()
                    base_thresh = mean_internal_sim + 0.05
                else:
                    base_thresh = 0.85
                    
                inferred_thresh = max(0.50, min(0.95, base_thresh - (usage_ratio ** 2) * 0.20))
                
                if best_sim.item() < inferred_thresh:
                    if len(empty_ids) > 0:
                        target_id = empty_ids[0]
                    else:
                        if self.consolidate(c_idx, total_class, threshold=inferred_thresh):
                            empty_ids = [idx for idx in class_slot_ids if self.usage_counts[idx] == 0]
                            target_id = empty_ids[0]
                        else:
                            target_id = best_id
                            count = self.usage_counts[target_id].item()
                            gamma_i = base_gamma / (count + 1.0)
                else:
                    target_id = best_id
                    count = self.usage_counts[target_id].item()
                    gamma_i = base_gamma / (count + 1.0)
            else:
                target_id = empty_ids[0]
            
            target_ids.append(target_id)
            dynamic_gammas.append(gamma_i)
            self.usage_counts[target_id] += 1
        
        for t_id in target_ids:
            self.usage_counts[t_id] -= 1
            
        t_ids = torch.tensor(target_ids, dtype=torch.long, device=values.device)
        t_gammas = torch.tensor(dynamic_gammas, dtype=values.dtype, device=values.device)
        self.memorize_batch(values, t_ids, t_gammas)

    @torch.no_grad()
    def auto_memorize_batch_fast(
        self,
        class_indices: Union[int, torch.Tensor],
        total_class: int,
        values: torch.Tensor
    ):
        batch_size = values.size(0)
        device = values.device

        if isinstance(class_indices, int):
            class_indices = torch.full((batch_size,), class_indices, dtype=torch.long, device=device)

        target_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        dynamic_gammas = torch.ones(batch_size, dtype=values.dtype, device=device)

        slots_per_class = self.capacity // total_class
        unique_classes = torch.unique(class_indices)

        for c_idx in unique_classes:
            c_mask = (class_indices == c_idx)
            c_values = values[c_mask]
            num_c_samples = c_values.size(0)

            start_id = c_idx * slots_per_class
            end_id = start_id + slots_per_class
            class_slot_ids = torch.arange(start_id, end_id, device=self.usage_counts.device)

            c_usage = self.usage_counts[class_slot_ids]
            used_mask = c_usage > 0
            used_slots = class_slot_ids[used_mask]
            empty_slots = class_slot_ids[~used_mask]

            if len(used_slots) == 0:
                target_ids[c_mask] = empty_slots[0]
                self.usage_counts[empty_slots[0]] += num_c_samples
                continue

            # 2. 向量化 Recall
            used_keys = self.K[used_slots] 
            stored_values = torch.matmul(self.W, used_keys.t()).t()
            
            counts = self.usage_counts[used_slots].unsqueeze(1).float()
            stored_values = stored_values / counts

            # 3. 向量化相似度计算
            val_norm = torch.nn.functional.normalize(c_values, p=2, dim=1)
            stored_norm = torch.nn.functional.normalize(stored_values, p=2, dim=1)
            sim = torch.matmul(val_norm, stored_norm.t()) 

            best_sims, best_local_indices = torch.max(sim, dim=1)
            best_slot_ids = used_slots[best_local_indices]

            # 4. 向量化动态阈值
            if len(used_slots) >= 2:
                internal_sim = torch.matmul(stored_norm, stored_norm.t())
                mask_off_diag = ~torch.eye(len(used_slots), dtype=torch.bool, device=device)
                base_thresh = internal_sim[mask_off_diag].mean().item() + 0.05
            else:
                base_thresh = 0.85

            usage_ratio = len(used_slots) / slots_per_class
            inferred_thresh = max(0.50, min(0.95, base_thresh - (usage_ratio ** 2) * 0.20))

            need_new_mask = best_sims < inferred_thresh
            final_c_target_ids = best_slot_ids.clone()

            empty_idx = 0
            for i in range(num_c_samples):
                if need_new_mask[i] and empty_idx < len(empty_slots):
                    final_c_target_ids[i] = empty_slots[empty_idx]
                    empty_idx += 1
                
                self.usage_counts[final_c_target_ids[i]] += 1

            target_ids[c_mask] = final_c_target_ids

        for t_id in target_ids:
            self.usage_counts[t_id] -= 1

        self.memorize_batch(values, target_ids, dynamic_gammas)

    def consolidate(self, class_index: int, total_class: int, threshold: float) -> bool:
        '''
        离线记忆压缩
        寻找该类中最相似的两个槽位，若其相似度达到 threshold，则融合并腾出空位。
        '''
        slots_per_class = self.capacity // total_class
        start_id = class_index * slots_per_class
        end_id = start_id + slots_per_class
        
        used_ids = [i for i in range(start_id, end_id) if self.usage_counts[i] > 0]
        
        if len(used_ids) < 2:
            return False 
            
        stored_values = torch.stack([self.recall(i).value for i in used_ids])
        
        norms = stored_values / (stored_values.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = torch.matmul(norms, norms.t())
        sim_matrix.fill_diagonal_(-float('inf'))
        
        max_idx = torch.argmax(sim_matrix)
        idx_A = (max_idx // len(used_ids)).item()
        idx_B = (max_idx % len(used_ids)).item()
        max_sim = sim_matrix[idx_A, idx_B].item()
        
        if max_sim < threshold:
            return False
            
        id_A, id_B = used_ids[idx_A], used_ids[idx_B]
        
        count_A = self.usage_counts[id_A].item()
        count_B = self.usage_counts[id_B].item()
        val_A = stored_values[idx_A]
        val_B = stored_values[idx_B]
        
        val_merged = (val_A * count_A + val_B * count_B) / (count_A + count_B)
        
        key_A, key_B = self.K[id_A], self.K[id_B]
        self.W -= torch.outer(val_A, key_A)
        self.W -= torch.outer(val_B, key_B)
        self.W += torch.outer(val_merged, key_A)
        
        self.usage_counts[id_A] = count_A + count_B
        self.usage_counts[id_B] = 0
        
        return True
    
    def replay(
        self,
        total_class: int,
        repeat: int
    ) -> CodonDataset:
        '''
        生成睡眠阶段的重放数据集
        遍历所有已使用的记忆槽位，根据槽位索引推断类别标签，
        通过 recall 提纯特征，并按照 repeat 次数重复扩充数据集。
        '''
        replayed_data = []
        slots_per_class = self.capacity // total_class
        
        valid_ids = torch.nonzero(self.usage_counts > 0).squeeze(-1).tolist()
        
        for slot_id in valid_ids:
            episode = self.recall(slot_id)
            class_label = slot_id // slots_per_class
            replayed_data.append((episode.value.detach(), class_label))
            
        repeated_data = replayed_data * repeat
        
        return EABReplayDataset(repeated_data)
    
    def count_params(
        self, 
        trainable_only: bool = False, 
        active_only: bool = False, 
        human_readable: bool = False, 
        seen: set = None
    ) -> Union[int, str]:
        '''
        重写参数统计方法。
        由于 EAB 的核心“权重” (W, K) 被注册为无梯度的 Buffer 以执行外积叠加，
        常规的参数统计会忽略它们。此方法将其正确计入，并支持动态的 active_only 计算。
        '''
        if seen is None:
            seen = set()

        # 1. 先调用父类方法，获取可能的标准 Parameter 计数（以防未来在块内添加了归一化层等）
        total = super().count_params(
            trainable_only=trainable_only, 
            active_only=active_only, 
            human_readable=False, 
            seen=seen
        )

        if not trainable_only:
            if self.W not in seen:
                seen.add(self.W)
                total += self.W.numel()
                
            if self.K not in seen:
                seen.add(self.K)
                seen.add(self.usage_counts)
                
                if active_only:
                    used_slots = int((self.usage_counts > 0).sum().item())
                    total += used_slots * self.key_dim
                    total += used_slots
                else:
                    total += self.K.numel()
                    total += self.usage_counts.numel()

        if human_readable:
            if total >= 1e9:
                return f'{total / 1e9:.2f}B'
            elif total >= 1e6:
                return f'{total / 1e6:.2f}M'
            elif total >= 1e3:
                return f'{total / 1e3:.2f}K'
            return str(total)

        return total