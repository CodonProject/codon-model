'''
LoopedVLM 接入 pretrain pipeline 的 8GB 显存实测。

要点：
  1. pretrain 的 ChunkedTokenStream 只会产出 (inputs, labels) 纯文本张量，
     图像没有通路 —— 所以多模态要自带 runner；
  2. PretrainConfig.batch_builder 负责把一条 batch 拆成 (模型 kwargs, labels)，
     模型侧只需 kwargs 里带上 images / image_patch_id；
  3. LoopedVLM 支持 image_patch_id：自己从 input_ids 里扫占位符位置，
     不必外部再传 image_patch_indices。
'''
import os, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from exp_loop import LoopedConfig, LoopedVLM
from codon.pipeline.pretrain import PretrainConfig, PretrainPipeline
from demo_image_embed import build_tokenizer


# ---------------------------------------------------------------- 多模态数据
class ToyVLMStream:
    '''按 plan 的 stage 产出 (stage, input_ids, labels, images)。

    序列结构：[文本 × pre][占位符 × P][文本 × post]，labels 与 input_ids 错位一格。
    '''

    def __init__(self, plan, vocab_size, patch_id, patches_per_image,
                 num_images_per_sample=1, image_size=224, seed=0):
        self.plan = plan
        self.vocab_size = vocab_size
        self.patch_id = patch_id
        self.patches_per_image = patches_per_image
        self.num_images = num_images_per_sample
        self.image_size = image_size
        self.seed = seed
        self.nested_images = False     # True -> 每 batch 行各给 num_images 张图
        self.stage_idx = 0
        self.step_in_stage = 0
        self._epoch = 0

    def __iter__(self):
        # 可重复迭代：每次重新迭代都从头开始（数据由 seed 决定，不依赖状态）
        self.stage_idx = 0
        self.step_in_stage = 0
        while self.stage_idx < len(self.plan.stages):
            stage = self.plan.stages[self.stage_idx]
            while self.step_in_stage < stage.steps:
                g = torch.Generator().manual_seed(self.seed + self.step_in_stage * 7919)
                b, l = stage.batch_size, stage.seq_len + 1
                patch_tokens = self.patches_per_image * self.num_images
                text_slots = l - patch_tokens
                if text_slots < 2:
                    raise ValueError(
                        f'seq_len={l} 装不下 {patch_tokens} 个占位符 + 文本')
                ids = torch.randint(4, self.vocab_size, (b, l), generator=g)
                start = max(1, text_slots // 3)
                ids[:, start:start + patch_tokens] = self.patch_id
                labels = ids.clone()
                # nested_images=False: 扁平 list，模型按 [images]*B 广播（每行同样的图）
                # nested_images=True : 每 batch 行各自 num_images 张图
                if self.nested_images:
                    images = [
                        [torch.randn(3, self.image_size, self.image_size, generator=g)
                         for _ in range(self.num_images)]
                        for _ in range(b)
                    ]
                else:
                    images = [
                        torch.randn(3, self.image_size, self.image_size, generator=g)
                        for _ in range(self.num_images)
                    ]
                yield stage, ids[:, :-1].contiguous(), labels[:, 1:].contiguous(), images
                self.step_in_stage += 1
            self.stage_idx += 1
            self.step_in_stage = 0

    def state_dict(self):
        return {'stage_idx': self.stage_idx, 'step_in_stage': self.step_in_stage}

    def load_state_dict(self, state):
        self.stage_idx = state.get('stage_idx', 0)
        self.step_in_stage = state.get('step_in_stage', 0)


class VLMPretrainPipeline(PretrainPipeline):
    '''多模态版 pretrain：数据源自带图像，绕开 ChunkedTokenStream。

    `setup()` 仍走基类的计划/优化器/调度器构建（只是计划只用于报步数），
    `iterate_epochs()` 换成 VLM 流。续训时 runner state 不在 checkpoint 里，
    因为数据流由 seed 决定、可重建。
    '''

    def __init__(self, model, tokenizer, config, stream, device=None):
        super().__init__(model, tokenizer, config, device=device)
        self._stream = stream

    def _ensure_runner(self, dataset):
        self._runner = self._stream

    def iterate_epochs(self, dataset):
        while True:
            yield self._stream

    def _apply_payload(self, payload):
        payload = dict(payload)
        payload['runner'] = None          # 多模态流不需要/不支持 runner state
        super()._apply_payload(payload)


def build_vlm(model_dim=128, num_heads=4, layers=1, vocab_size=512, image_size=224):
    cfg = LoopedConfig(
        model_dim=model_dim, num_heads=num_heads, recurrence=1,
        prelude_attn_types=['mha'],
        body_attn_types=['mha', 'gdn'] * layers,
        coda_attn_types=['mha'],
        pos_emb_type='fourier_interleaved', pos_num_axes=2,
    )
    return LoopedVLM(cfg, vocab_size=vocab_size, backend='small', image_size=image_size)


def main():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', end='')
    if device.type == 'cuda':
        print(f' | vram={torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB')
    else:
        print()

    PATCH_ID = 1
    PATCHES = 49                     # 224/32 -> 7x7
    ckpt_dir = './_tmp_vlm_ckpt'
    shutil.rmtree(ckpt_dir, ignore_errors=True)   # 冒烟测试：不续训，避免读旧 checkpoint
    model = build_vlm()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_vision = sum(p.numel() for p in model.vision.parameters())
    print(f'params total={model.count_params(human_readable=True)} | trainable={n_train/1e6:.2f}M '
          f'| vision frozen={not model.vision.conv1.weight.requires_grad} ({n_vision/1e6:.2f}M)')

    cfg = PretrainConfig(
        compiled=False,
        base_context=64,
        target_context=64,
        global_batch_tokens=64 * 2,
        step_mode='min',
        use_progress=False,
        save_every_steps=0,
        sanity_every_steps=0,
        ckpt_dir=ckpt_dir,
        batch_builder=lambda b: (
            {'input_ids': b[1], 'images': b[3], 'image_patch_id': PATCH_ID},
            b[2],
        ),
    )
    tokenizer = build_tokenizer()

    # 计划里 stage 的 seq_len 由 config 决定；数据流按计划构造
    from codon.utils.plan import ContextTrainingPlanner
    planner = ContextTrainingPlanner(
        model, step_mode='min', base_context=64, target_context=64, global_batch_tokens=128)
    plan = planner.generate_plan()
    plan.stages = plan.stages[:1]        # 只跑 Foundation
    stream = ToyVLMStream(plan, vocab_size=512, patch_id=PATCH_ID, patches_per_image=PATCHES)

    pipe = VLMPretrainPipeline(model, tokenizer=tokenizer, config=cfg,
                               stream=stream, device=device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    metrics = pipe.train(stream, num_epochs=1, steps_per_epoch=3,
                         batch_fn=lambda s: iter(s))
    dt = time.time() - t0

    print('metrics:', {k: round(v, 4) for k, v in metrics.items()})
    print(f'steps=3 | {dt:.1f}s')
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f'peak allocated={peak:.2f}GB | peak reserved={reserved:.2f}GB '
              f'| 8GB 余量={7.96 - reserved:.2f}GB')
    print('OK')


if __name__ == '__main__':
    main()
