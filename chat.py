from codon.motif import MotifA1

model = MotifA1().load_pretrained('a1_sft.safetensors').to('cuda')

print(model.count_params(human_readable=True))

from codon.utils.tokens import PackedTokenizer

tokenizer = PackedTokenizer('motif.vocab')

from codon.utils.generate import chat
from rich.console import Console

console = Console()

for chunk in chat(
    model, tokenizer, model.device, messages=[{'role': 'user', 'content': '学校是监狱吗？'}], stream=True, max_new_tokens=1024
    ):
    if chunk.cot_ended: console.print('\n')
    if chunk.is_cot:
        console.print(chunk.content, end='', style='blue')
    else:
        console.print(chunk.content, end='')