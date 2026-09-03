from codon import *
from codon.motif.motif_a2 import MotifA2
from codon.utils.tokens import PackedTokenizer

model = MotifA2().load('./dev/base.safetensors').to_device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = PackedTokenizer('./tokenizer.zip')

from codon.pipeline.sft import SFTConfig, SFTStage, SFTPipeline, build_sft_stages

pipeline = SFTPipeline(
    model=model,
    tokenizer=tokenizer
)