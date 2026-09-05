from codon import *
from codon.motif.motif_a2 import MotifA2
from codon.utils.tokens import PackedTokenizer

model = MotifA2().load('./dev/base.safetensors').to_device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = PackedTokenizer('./tokenizer.zip')

from codon.pipeline.sft import SFTConfig, SFTPipeline

config = SFTConfig(
    learning_rate=3e-4,
    lora=True,
    ckpt_dir='./sft_ckpt',
    probe_every_steps=500,
    stage_specs=[
        {'name': 'stage1', 'folder': 'D:/Datasets/Role', 'epochs': 1, 'ckpt': 'a2_sft_stage1_adapter.safetensors'},
    ],
    pad_length=1024,
    batch_size=3
)

pipeline = SFTPipeline(
    model=model,
    tokenizer=tokenizer,
    config=config,
)

metrics = pipeline.train()
