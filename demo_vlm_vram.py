'''
8GB 显存扫描：LoopedVLM 走 pretrain pipeline 的真实峰值显存。

每个配置跑 2 步（含一次 optimizer.step），记录 torch.cuda.max_memory_allocated。
'''
import os, sys, gc, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from exp_loop import LoopedConfig, LoopedVLM
from codon.pipeline.pretrain import PretrainConfig
from codon.utils.plan import TrainingPlan, Stage
from demo_vlm_pretrain import ToyVLMStream, VLMPretrainPipeline
from demo_image_embed import build_tokenizer

DEVICE = torch.device('cuda')
LIMIT = 7.96
PATCH_ID = 1
PATCHES = 49
VOCAB = 4096


def run_case(name, model_dim, num_heads, layers, seq_len, batch_size,
             num_images, image_size=224, body=None, grad_ckpt=False):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    body = body or ['mha', 'gdn'] * layers
    cfg = LoopedConfig(model_dim=model_dim, num_heads=num_heads, recurrence=1,
                       prelude_attn_types=['mha'], body_attn_types=body,
                       coda_attn_types=['mha'],
                       pos_emb_type='fourier_interleaved', pos_num_axes=2)
    model = LoopedVLM(cfg, vocab_size=VOCAB, backend='small', image_size=image_size)
    if grad_ckpt:
        model.gradient_checkpointing = True
    model = model.to(DEVICE)

    plan = TrainingPlan(total_tokens=0, total_steps=2, step_mode='min', stages=[
        Stage(name='F', seq_len=seq_len, chunk_len=0, batch_size=batch_size, tokens=0, steps=2)
    ])
    stream = ToyVLMStream(plan, vocab_size=VOCAB, patch_id=PATCH_ID,
                          patches_per_image=PATCHES, num_images_per_sample=num_images,
                          image_size=image_size)
    stream.nested_images = True       # 每 batch 行各自持有图像，避免 [images]*B 广播
    ckpt_dir = f'./_tmp_sw_{abs(hash(name)) % 10000}'
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    pcfg = PretrainConfig(
        compiled=False, base_context=seq_len, target_context=seq_len,
        global_batch_tokens=seq_len * batch_size, step_mode='min',
        use_progress=False, save_every_steps=0, sanity_every_steps=0, ckpt_dir=ckpt_dir,
        batch_builder=lambda b: (
            {'input_ids': b[1], 'images': b[3], 'image_patch_id': PATCH_ID}, b[2]),
    )
    pipe = VLMPretrainPipeline(model, build_tokenizer(), pcfg, stream, device=DEVICE)

    t0 = time.time()
    try:
        pipe.train(stream, num_epochs=1, steps_per_epoch=2, batch_fn=lambda s: iter(s))
        ok = True
    except torch.cuda.OutOfMemoryError:
        ok = False
    dt = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**3
    status = 'OK ' if ok else 'OOM'
    print(f'{status} | {name:<42} peak={peak:5.2f}GB  {dt:5.1f}s  '
          f'{"余量 %.2fGB" % (LIMIT - peak) if ok else ""}')
    del model, pipe
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    gc.collect(); torch.cuda.empty_cache()
    return ok, peak


def main():
    print(f'device={torch.cuda.get_device_name(0)} vram={LIMIT}GB')
    print('-' * 88)
    # 命名: dim/heads/层数 x seq x bs x 图数
    run_case('d128 h4 L2 | seq256 bs8 img1', 128, 4, 2, 256, 8, 1)
    run_case('d256 h4 L2 | seq256 bs8 img1', 256, 4, 2, 256, 8, 1)
    run_case('d384 h6 L2 | seq256 bs8 img1', 384, 6, 2, 256, 8, 1)
    run_case('d512 h8 L2 | seq256 bs8 img1', 512, 8, 2, 256, 8, 1)
    run_case('d512 h8 L2 | seq512 bs4 img1', 512, 8, 2, 512, 4, 1)
    run_case('d512 h8 L2 | seq512 bs4 img2', 512, 8, 2, 512, 4, 2)
    run_case('d768 h12 L2| seq256 bs4 img1', 768, 12, 2, 256, 4, 1)
    run_case('d768 h12 L2| seq512 bs2 img1', 768, 12, 2, 512, 2, 1)
    run_case('d768 h12 L2| seq512 bs4 img1', 768, 12, 2, 512, 4, 1)
    run_case('d768 h12 L4| seq256 bs4 img1', 768, 12, 4, 256, 4, 1)
    run_case('d1024 h16 L2|seq256 bs2 img1', 1024, 16, 2, 256, 2, 1)
    print('-' * 88)
    print('注：图像固定 224x224（MobileNetV3 特征图 7x7=49 patch）；'
          'recurrence=1；未开 torch.compile。')


if __name__ == '__main__':
    main()
