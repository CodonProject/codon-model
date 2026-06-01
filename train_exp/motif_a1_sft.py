# sft_a1.py
from codon.utils.seed import seed_everything

seed_everything(seed=42, verbose=False)

from codon.utils.tokens   import PackedTokenizer
from codon.kit.train      import run_chat_turn
from codon.motif.data     import MotifSFT
from codon.motif          import MotifA1
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------- Tokenizer & data ----------
tokenizer = PackedTokenizer('./motif.vocab')

stages = [
    {'name': 'stage1', 'folder': './sft',   'epochs': 1, 'ckpt': 'a1_sft_stage1.safetensors'},
    {'name': 'stage2', 'folder': './sft_2', 'epochs': 1, 'ckpt': 'a1_sft_stage2.safetensors'}
]

datasets = [
    MotifSFT(
        folder=s['folder'],
        tokenizer=tokenizer,
        pad_length=2048,
        batch_size=8,
    )
    for s in stages
]

# ---------- Model ----------
device = 'cuda'
model = MotifA1().load_pretrained('a1.safetensors').to(device)
model.train()

# ---------- Schedule (unified across stages) ----------
stage_steps  = [len(ds) * s['epochs'] for ds, s in zip(datasets, stages)]
total_steps  = sum(stage_steps)
warmup_steps = 100
sample_every = 2000

peak_lr        = 5e-5
warmup_base_lr = 1e-8
eta_min        = peak_lr * 0.1

optimizer = AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=warmup_base_lr / peak_lr,
    end_factor=1.0,
    total_iters=warmup_steps,
)
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max(1, total_steps - warmup_steps),
    eta_min=eta_min,
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps],
)

# ---------- Chat probes ----------
probe_prompts = [
    '用一句话解释什么是注意力机制。',
    'Translate to English: 今天天气真好。',
    'Write a haiku about the ocean.',
]

# ---------- Training ----------
pbar = tqdm(total=total_steps, desc='SFT Training', dynamic_ncols=True)
global_step = 0

for stage_idx, (stage, dataset) in enumerate(zip(stages, datasets)):
    pbar.write(f'\n##### Entering {stage["name"]} '
               f'(folder={stage["folder"]}, epochs={stage["epochs"]}, '
               f'batches/epoch={len(dataset)}) #####')

    for epoch in range(stage['epochs']):
        for i in range(len(dataset)):
            data = dataset[i]

            input_ids = data['input_ids'].to(device)
            labels    = data['labels'].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(input_ids)

                shift_logits = output.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            global_step += 1
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'Stage': stage['name'],
                'Ep':    f'{epoch + 1}/{stage["epochs"]}',
                'Loss':  f'{loss.item():.4f}',
                'LR':    f'{current_lr:.2e}',
            })
            pbar.update(1)

            if global_step % sample_every == 0:
                pbar.write(f'\n===== [Chat Probe @ step {global_step} / {stage["name"]}] =====')
                for prompt in probe_prompts:
                    run_chat_turn(
                        model, tokenizer, device,
                        step=global_step,
                        user_prompt=prompt,
                        max_new_tokens=128,
                        temperature=0.8,
                    )
                pbar.write('=' * 56)

    pbar.write(f'[*] Saving checkpoint for {stage["name"]} -> {stage["ckpt"]}')
    model.save_pretrained(stage['ckpt'])

pbar.close()
print('[*] SFT complete. Final model saved to', stages[-1]['ckpt'])