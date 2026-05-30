from codon.motif import MotifA1

model = MotifA1().load_pretrained('a1_sft.safetensors').to('cuda')

print(model.count_params(human_readable=True))

from codon.utils.tokens import PackedTokenizer

tokenizer = PackedTokenizer('motif.vocab')

from codon.utils.generate import chat
from rich.console import Console

console = Console()

messages = [
    {'role': 'user', 'content': '什么是人类'}
]

console.print(f"\n[bold yellow]User:[/bold yellow] {messages[0]['content']}\n")
console.print("[bold magenta]Model Thinking...[/bold magenta]")

try:
    for chunk in chat(
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        messages=messages,
        max_new_tokens=1024,
        temperature=0.3
    ):
        if chunk.cot_ended: 
            console.print('\n\n[bold green]答：[/bold green]', end='')
            
        if chunk.is_cot:
            console.print(chunk.content, end='', style='blue')
        else:
            console.print(chunk.content, end='')
    console.print()
except KeyboardInterrupt:
    console.print("\n[red]生成被用户中断。[/red]")