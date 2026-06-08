from codon.utils.seed import seed_everything

seed_everything(seed=42, verbose=False)

from codon.motif import CausalLanguageModelOutput, MotifA1Tokenizer
from codon.motif.config.a1 import build_config, build_optim_and_scheduler
from codon.utils.plan import ContextTrainingPlanner, StatefulPlanRunner
from codon.motif.data import MotifPrev1
from codon.utils.tokens import PackedTokenizer
from codon.utils.lifecycle import register_exit
from codon.kit.train import run_sanity_check
from tqdm import tqdm
import torch.nn.functional as F
import torch, os

from mods import Model

from teleboard import TeleBoard

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

test = 'mha'

board = TeleBoard(
    'https://teleboard.codon.top',
    send_interval=10
)

writer = board.open_exp(f'Tans/{test}')

model = Model(test)

print('[*] Running torch.compile (Dynamic=True)...')

compiled_model = torch.compile(model, dynamic=True) 

plan = ContextTrainingPlanner(
    model,
    target_context=4096,
    global_batch_tokens=8192*2
).generate_plan().print_report()

config = build_config(plan.total_steps)
optimizer, scheduler = build_optim_and_scheduler(model, config)

tokenizer = MotifA1Tokenizer().from_remote()
dataset   = MotifPrev1('./prev1').set_tokenizer(tokenizer)
eos_id    = tokenizer.fast_tokenizer.eos_token_id

runner = StatefulPlanRunner(plan, dataset, eos_id)

@register_exit
def save(name: str = f'{test}_last'):
    checkpoint = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'sched': scheduler.state_dict(),
        'runner': runner.state_dict()
    }
    torch.save(checkpoint, f'{name}.pt')
    tqdm.write(f'[*] Checkpoint saved to {name}.pt')

def load(name: str = f'{test}_last'):
    file_path = f'{name}.pt'
    if not os.path.exists(file_path):
        print(f'[*] No checkpoint found at {file_path}, starting from scratch.')
        return
    
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optim'])
    scheduler.load_state_dict(checkpoint['sched'])
    runner.load_state_dict(checkpoint['runner'])
    print(f'[*] Successfully loaded checkpoint from {file_path}')

def get_global_step() -> int:
    step = 0
    for i in range(runner.current_stage_idx):
        step += runner.plan.stages[i].steps
    step += runner.step_within_stage
    return step

load(f'{test}_last')

device = model.to('cuda').device
model.train()

global_step = sum(plan.stages[i].steps for i in range(runner.current_stage_idx)) + runner.step_within_stage

run_sanity_check(model, tokenizer, device, eos_id, global_step)

pbar = tqdm(
    total=plan.total_steps, 
    initial=global_step, 
    desc='Pretraining',
    dynamic_ncols=True
)

for stage_info, inputs, labels in runner:
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output: CausalLanguageModelOutput = compiled_model(inputs) 
        logits = output.logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_norm'])
    
    optimizer.step()
    scheduler.step()
    
    global_step += 1
    current_lr = scheduler.get_last_lr()[0]
    
    pbar.set_postfix({
        'Stage': stage_info.name,
        'Seq': stage_info.seq_len,
        'Loss': f'{loss.item():.4f}',
        'LR': f'{current_lr:.2e}'
    })
    pbar.update(1)

    writer.add_scalar('loss', loss.item(), global_step)
    writer.add_scalar('LR', current_lr, global_step)
        
    if global_step % 2000 == 0:
        save(f'{test}_last')
        run_sanity_check(model, tokenizer, device, eos_id, global_step)
    
pbar.close()
board.close()
save(f'{test}_final')