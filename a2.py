from codon import *
from codon.motif.motif_a2 import MotifA2

print('[*] Building model...')
model = MotifA2().to_device('cuda' if torch.cuda.is_available() else 'cpu').compiled(dynamic=True)

from codon.utils.data.text import TextFileDataset, PackedTokenizer

tokeniz = PackedTokenizer('./data/tokenizer.zip')
dataset = TextFileDataset('./output.parquet', recursive=True, tokenizer=tokeniz, seq_len=8192, drop_last=False, shuffle=True, seed=42)
# dataloader = dataset.compose().loader()
eos_id = tokeniz.token_to_id('<|im_end|>')
total_steps = len(dataset)

from codon.optim import GroupOptimizer

opt_groups = model.optimizer_groups(
    weight_decay=0.05,
    base_lr=6e-4,
    adamw_kwargs={
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
)
optimizer = GroupOptimizer.build(
    opt_groups=opt_groups,
    unified_scheduler={
        'warmup_steps': 200,
        'total_steps': total_steps * 1.3,
        'lr_min': 6e-5,
        'start_factor': 1e-8,
    }
)

current_step = 0
total_updates = 0
loss_accum = 0.0
count_accum = 0

from codon.utils.lifecycle import exit_manager
from tqdm import tqdm
import os

@exit_manager.register
def save(name: str = 'last'):
    checkpoint = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'step': current_step,
        'seg': total_updates,
    }
    model.save('./base.safetensors')
    torch.save(checkpoint, f'{name}.pt')
    torch.save(checkpoint, f'./data/nsfw_base.pt')
    tqdm.write(f'[*] Checkpoint saved to {name}.pt')

def load(name: str = 'last'):
    global current_step, total_updates
    file_path = f'{name}.pt'
    if not os.path.exists(file_path):
        tqdm.write(f'[*] No checkpoint found at {file_path}, starting from scratch.')
        return
    
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optim'])
    current_step = checkpoint['step']
    total_updates = checkpoint['seg']
    tqdm.write(f'[*] Successfully loaded checkpoint from {file_path}')

from codon.kit.train import run_sanity_check
from teleboard import TeleWriter
import time

load('last')

writer = TeleWriter(
    exp_id='nsfw_model',
    server_url='https://teleboard.codon.top',
    api_key='tele'
)

generated_text = run_sanity_check(model, tokeniz, model.device, eos_id, current_step, '这时')
writer.add_text('Sanity/Init', generated_text, total_updates)

gradient_accumulation_steps = 16
accum_step_counter = 0
last_time = time.time()

pbar = tqdm(dataset, desc='Training', total=total_steps, initial=current_step)
for batch_idx, batch in enumerate(pbar):
    length = len(batch)
    for idx in range(length):
        seq = batch[idx]
        seq = torch.tensor(seq).to(model.device)
        inputs = seq[:-1].unsqueeze(0)
        labels = seq[1:].unsqueeze(0)
    
        with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
            output = model(inputs) 
            logits = output.logits
            aux = output.aux_loss
            aux = torch.tensor(0.0) if aux is None else aux
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1)) + aux

        loss.backward()
        accum_step_counter += 1
        loss_accum += loss.item()
        count_accum += 1

        del output, logits

        if accum_step_counter % gradient_accumulation_steps == 0:
            torch.cuda.empty_cache()
            total_updates += 1
            avg_loss = loss_accum / count_accum if count_accum > 0 else 0.0
            total_norm = model.clip_grad_norm('auto')
            optimizer.step()
            optimizer.step_schedulers()
            optimizer.zero_grad()
            accum_step_counter = 0
            loss_accum = 0.0
            count_accum = 0
            
            all_lrs = optimizer.get_all_lr()
            current_lr = []
            for lr in all_lrs.values():
                if isinstance(lr, list):
                    current_lr.extend(lr)
                else:
                    current_lr.append(lr)
            if isinstance(current_lr, (list, tuple)):
                current_lr = sum(current_lr) / len(current_lr)

            writer.add_scalar('Loss/train_avg', avg_loss, total_updates)
            writer.add_scalar('Loss/aux', aux.item(), total_updates)
            writer.add_scalar('LR', current_lr, total_updates)
            writer.add_scalar('Grad_norm', total_norm, total_updates)
            now = time.time()
            throughput = 1.0 / (now - last_time) if (now - last_time) > 0 else 0
            writer.add_scalar('Throughput', throughput, total_updates)
            last_time = now
            writer.add_scalar('Progress/percent', current_step / total_steps * 100, total_updates)
        else:
            all_lrs = optimizer.get_all_lr()
            current_lr = []
            for lr in all_lrs.values():
                if isinstance(lr, list):
                    current_lr.extend(lr)
                else:
                    current_lr.append(lr)
            if isinstance(current_lr, (list, tuple)):
                current_lr = sum(current_lr) / len(current_lr)

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'LR': f'{current_lr:.2e}',
            'Seg': f'{idx + 1}/{length}',
            'Acc': f'{accum_step_counter + 1}/{gradient_accumulation_steps}',
            'AUX': f'{aux.item():.4f}'
        })

    current_step += 1
    pbar.update()

    if current_step % 500 == 0:
        save('last')
        generated_text = run_sanity_check(model, tokeniz, model.device, eos_id, current_step, '这时')
        writer.add_text('Sanity/Check', generated_text, total_updates)

if accum_step_counter > 0:
    avg_loss = loss_accum / count_accum if count_accum > 0 else 0.0
    total_norm = model.clip_grad_norm('auto')
    optimizer.step()
    optimizer.step_schedulers()
    optimizer.zero_grad()
    all_lrs = optimizer.get_all_lr()
    current_lr = []
    for lr in all_lrs.values():
        if isinstance(lr, list):
            current_lr.extend(lr)
        else:
            current_lr.append(lr)
    if isinstance(current_lr, (list, tuple)):
        current_lr = sum(current_lr) / len(current_lr)
    writer.add_scalar('Loss/train_avg', avg_loss, total_updates)
    writer.add_scalar('LR', current_lr, total_updates)
    writer.add_scalar('Grad_norm', total_norm, total_updates)
    tqdm.write('[*] Final gradient step performed.')

writer.close()
tqdm.write('[*] Training finished.')