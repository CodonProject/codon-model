import math
import numpy as np
from codon.utils.data import ChunkedTokenStream, CodonDataset, Stateful
from codon.base import BasicModel
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Iterable, Optional

@dataclass
class TrainingStepsConfig:
    tokens_per_step: int
    model_size: int
    
    min_tokens: int
    recommended_tokens: int
    overtrain_tokens: int
    
    min_steps: int
    recommended_steps: int
    overfit_steps: int

def calculate_training_steps(
    model_size: int, 
    tokens_per_sample: int, 
    batch_size: int,
    min_tpp: float = 20.0,
    rec_tpp: float = 80.0,
    overfit_tpp: float = 200.0
) -> TrainingStepsConfig:
    tokens_per_step = tokens_per_sample * batch_size
    
    min_tokens = int(model_size * min_tpp)
    recommended_tokens = int(model_size * rec_tpp)
    overtrain_tokens = int(model_size * overfit_tpp)
    
    return TrainingStepsConfig(
        tokens_per_step=tokens_per_step,
        model_size=model_size,
        min_tokens=min_tokens,
        recommended_tokens=recommended_tokens,
        overtrain_tokens=overtrain_tokens,
        min_steps=math.ceil(min_tokens / tokens_per_step),
        recommended_steps=math.ceil(recommended_tokens / tokens_per_step),
        overfit_steps=math.ceil(overtrain_tokens / tokens_per_step)
    )


@dataclass
class ContextTrainingParams:
    base_context: int = 1024
    total_token_budget: float = 100.0
    data_distribution_factor: float = 0.5
    performance_tradeoff_preference: float = 0.5

def calculate_training_schedule(params: ContextTrainingParams, target_len: int) -> Dict[str, Any]:
    foundation_weight = 0.4 
    
    length_ratio = target_len / params.base_context
    log_ratio = np.log(length_ratio) if length_ratio > 1 else 0.0
    budget_sensitivity = 1.0 - (1.0 / (1.0 + params.total_token_budget / 50.0))
    
    expansion_weight = (log_ratio * 0.2) * budget_sensitivity
    
    stabilization_weight = (
        (params.performance_tradeoff_preference * 0.3) + 
        (np.log2(max(target_len, 2)) / 14.0) * 0.3 + 
        (params.data_distribution_factor * 0.2) + 
        ((params.total_token_budget - 20.0) / 100.0) * 0.2
    )
    
    total_weight = foundation_weight + expansion_weight + stabilization_weight
    normalized_weights = [
        foundation_weight / total_weight,
        expansion_weight / total_weight,
        max(0.0, stabilization_weight / total_weight)
    ]
    
    stages = [
        {
            'name': 'foundation', 
            'context_len': str(params.base_context), 
            'type': 'fixed',
            'allocated_tokens_B': params.total_token_budget * normalized_weights[0]
        },
        {
            'name': 'expansion', 
            'context_len': f'{params.base_context}-{target_len}', 
            'type': 'dynamic_increase',
            'allocated_tokens_B': params.total_token_budget * normalized_weights[1]
        },
        {
            'name': 'stabilization', 
            'context_len': str(target_len), 
            'type': 'fixed',
            'allocated_tokens_B': params.total_token_budget * normalized_weights[2]
        }
    ]
    
    summary = (
        f'For a {target_len} context goal with a {params.total_token_budget}B total budget, '
        f'the recommended training split is approximately:\n'
        f'  - {normalized_weights[0]:.1%} ({stages[0]["allocated_tokens_B"]:.1f}B) for Foundation (len {params.base_context})\n'
        f'  - {normalized_weights[1]:.1%} ({stages[1]["allocated_tokens_B"]:.1f}B) for Expansion ({params.base_context}->{target_len})\n'
        f'  - {normalized_weights[2]:.1%} ({stages[2]["allocated_tokens_B"]:.1f}B) for Stabilization (len {target_len})'
    )
    
    return {
        'input_parameters': params.__dict__,
        'target_context_length': target_len,
        'stages': stages,
        'weights_normalized': normalized_weights,
        'summary': summary
    }


@dataclass
class Stage:
    name: str
    sample_size: int
    batch_size: int
    tokens: int
    steps: int

    def build_stream(
        self, 
        data: Union[Iterable[Any], CodonDataset], 
        eos_token_id: int
    ) -> ChunkedTokenStream:
        return ChunkedTokenStream(
            data=data,
            chunk_len=self.chunk_len,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            eos_token_id=eos_token_id
        )

@dataclass
class TrainingPlan:
    total_tokens: int
    total_steps: int
    stages: List[Stage]
    step_mode: str

    def print_report(self) -> 'TrainingPlan':
        print('=' * 70)
        print(f' LLM Context Training Plan | Strategy: [{self.step_mode.upper()}]')
        print('=' * 70)
        print(f'Total Budget: {self.total_tokens / 1e9:,.3f} B Tokens | Total Steps: {self.total_steps:,}')
        print('-' * 70)
        
        for i, s in enumerate(self.stages):
            print(f'➜ [{i+1}] {s.name:<18} | Seq (sample_size)={s.sample_size:<5} | '
                  f'BS (batch_size)={s.batch_size:<4} | Tokens: {s.tokens/1e9:>5.3f}B | Steps: {s.steps:>8,}')
        
        print('=' * 70)
        return self


class ContextTrainingPlanner:
    def __init__(
        self, 
        model, 
        step_mode: str = 'recommended', # 'min', 'recommended', 'overfit'
        base_context: int = 512, 
        target_context: int = 8192,
        global_batch_tokens: int = 8192 * 2
    ):
        self.model = model
        self.step_mode = step_mode
        self.base_context = base_context
        self.target_context = target_context
        self.global_batch_tokens = global_batch_tokens
        
    def _get_expansion_lengths(self) -> List[int]:
        lengths = []
        curr = self.base_context * 2
        while curr < self.target_context:
            lengths.append(curr)
            curr *= 2
        return lengths

    def _create_stage(self, name: str, seq_len: int, target_tokens: int) -> Stage:
        effective_len = seq_len + 1
        
        batch_size = max(1, self.global_batch_tokens // effective_len)
        chunk_len = batch_size * effective_len
        
        steps = math.ceil(target_tokens / chunk_len)
        
        return Stage(
            name=name,
            seq_len=seq_len,
            batch_size=batch_size,
            chunk_len=chunk_len,
            tokens=target_tokens,
            steps=steps
        )

    def generate_plan(self) -> TrainingPlan:
        param_count = self.model.count_params(active_only=True) if isinstance(self.model, BasicModel) else sum(p.numel() for p in self.model.parameters())
        
        base_bs = max(1, self.global_batch_tokens // self.base_context)
        data = calculate_training_steps(param_count, self.base_context, base_bs)
        
        if self.step_mode == 'min':
            total_tokens = data.min_tokens
        elif self.step_mode == 'overfit':
            total_tokens = data.overtrain_tokens
        else:
            total_tokens = data.recommended_tokens
            
        ctp = ContextTrainingParams(base_context=self.base_context, total_token_budget=total_tokens / 1e9)
        schedule = calculate_training_schedule(ctp, target_len=self.target_context)
        stages_info = schedule['stages']
        stages: List[Stage] = []
        
        f_tokens = int(stages_info[0]['allocated_tokens_B'] * 1e9)
        stages.append(self._create_stage(
            name='Foundation', 
            sample_size=self.base_context, 
            target_tokens=f_tokens
        ))
        
        e_tokens_total = int(stages_info[1]['allocated_tokens_B'] * 1e9)
        exp_lengths = self._get_expansion_lengths()
        
        if exp_lengths:
            tokens_per_exp = e_tokens_total // len(exp_lengths)
            for i, c_len in enumerate(exp_lengths):
                stages.append(self._create_stage(
                    name=f'Expansion_{c_len}', 
                    sample_size=c_len, 
                    target_tokens=tokens_per_exp
                ))
                
        
        s_tokens = int(stages_info[2]['allocated_tokens_B'] * 1e9)
        stages.append(self._create_stage(
            name='Stabilization', 
            sample_size=self.target_context, 
            target_tokens=s_tokens
        ))

        total_steps = sum(stage.steps for stage in stages)
        
        return TrainingPlan(
            total_tokens=total_tokens,
            total_steps=total_steps,
            stages=stages,
            step_mode=self.step_mode
        )


class StatefulPlanRunner(Stateful):
    def __init__(self, plan: TrainingPlan, data: Any, eos_token_id: int):
        self.plan = plan
        self.data = data
        self.eos_token_id = eos_token_id
        
        self.current_stage_idx: int = 0
        self.step_within_stage: int = 0
        
        self._active_stream: Optional[ChunkedTokenStream] = None
        self._active_iterator = None

    def _init_stage_stream(self, stage_idx: int):
        stage = self.plan.stages[stage_idx]
        self._active_stream = stage.build_stream(self.data, self.eos_token_id)
        dataset_wrapper = self._active_stream.compose(seek=0)
        self._active_iterator = iter(dataset_wrapper)

    def state_dict(self) -> Dict[str, Any]:
        state = {
            'current_stage_idx': self.current_stage_idx,
            'step_within_stage': self.step_within_stage,
        }
        if self._active_stream is not None:
            state['stream_state'] = self._active_stream.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.current_stage_idx = state.get('current_stage_idx', 0)
        self.step_within_stage = state.get('step_within_stage', 0)
        
        if self.current_stage_idx < len(self.plan.stages):
            self._init_stage_stream(self.current_stage_idx)
            if 'stream_state' in state and self._active_stream is not None:
                self._active_stream.load_state_dict(state['stream_state'])
                dataset_wrapper = self._active_stream.compose(seek=0)
                self._active_iterator = iter(dataset_wrapper)

    def __iter__(self):
        if self._active_stream is None and self.current_stage_idx < len(self.plan.stages):
            self._init_stage_stream(self.current_stage_idx)

        while self.current_stage_idx < len(self.plan.stages):
            current_stage = self.plan.stages[self.current_stage_idx]
            
            while self.step_within_stage < current_stage.steps:
                try:
                    inputs, labels = next(self._active_iterator)
                    
                    yield current_stage, inputs, labels
                    
                    self.step_within_stage += 1
                except StopIteration:
                    self._init_stage_stream(self.current_stage_idx)
            
            self.current_stage_idx += 1
            self.step_within_stage = 0
            if self.current_stage_idx < len(self.plan.stages):
                self._init_stage_stream(self.current_stage_idx)