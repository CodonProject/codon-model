import os
import time
import numpy as np
import onnxruntime as ort
from rich.console import Console
from codon.motif import MotifA1Tokenizer

def main():
    console = Console()
    console.print("[bold yellow]==================================================[/bold yellow]")
    console.print("[bold yellow]🔋 Production-Ready Unified ONNX Streaming Chat CLI[/bold yellow]")
    console.print("[bold yellow]==================================================[/bold yellow]")

    unified_onnx = "motifa1.onnx"

    if not os.path.exists(unified_onnx):
        console.print("[bold red][Error][/bold red] ONNX unified model file not found! Please run export_motif_a1_remote.py first to export.")
        return

    # 1. Load SFT Tokenizer
    console.print("[bold magenta]Loading remote tokenizer...[/bold magenta]")
    tokenizer = MotifA1Tokenizer().from_remote()
    console.print("✓ Tokenizer successfully loaded!")

    # Retrieve special token IDs for parsing
    cot_start_id = tokenizer.token_to_id('[cot_start]')
    cot_end_id = tokenizer.token_to_id('[cot_end]')
    im_end_id = tokenizer.token_to_id('[im_end]')
    pad_id = tokenizer.token_to_id('[pad]')

    # 2. Initialize ONNX Session
    console.print("[bold magenta]Loading Unified ONNX Session with CPU/CUDA Execution Provider...[/bold magenta]")
    available_providers = ort.get_available_providers()
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    sess = ort.InferenceSession(unified_onnx, providers=providers)

    # Automatically deduce model configurations from inputs
    inputs = [node.name for node in sess.get_inputs()]
    past_key_names = [name for name in inputs if name.startswith("past_key_")]
    num_layers = len(past_key_names)

    # Find the shape metadata of past_key_0 to determine head params
    past_key_0_node = [node for node in sess.get_inputs() if node.name == "past_key_0"][0]
    past_key_shape = past_key_0_node.shape # e.g. [batch_size, num_kv_heads, past_seq_len, head_dim]
    
    # Normally [1, 2, 'past_seq_len', 96]
    # We fall back to 2 heads and 96 dim if shape contains dynamic symbols instead of static ints
    num_kv_heads = 2
    if isinstance(past_key_shape[1], int):
        num_kv_heads = past_key_shape[1]
    head_dim = 96
    if isinstance(past_key_shape[3], int):
        head_dim = past_key_shape[3]

    console.print(f"✓ Unified ONNX Session loaded (Layers: {num_layers}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}, Providers: {providers})")

    # 3. Interactive Loop
    messages = []
    
    console.print("\n[bold green]Ready! Type your prompt below. Type 'exit' or 'quit' to quit.[/bold green]")
    while True:
        try:
            user_input = console.input("\n[bold yellow]User:[/bold yellow] ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            messages.append({'role': 'user', 'content': user_input})
            
            # Format using standard chat template
            prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            prompt_len = len(prompt_ids)

            console.print("\n[bold magenta]Model thinking and generating...[/bold magenta]")

            # ====================================================
            # A. Prefill Stage (Process prompt with dynamic length 0 past caches)
            # ====================================================
            feeds = {}
            if "input_ids" in inputs:
                feeds["input_ids"] = np.array([prompt_ids], dtype=np.int64)
            if "start_pos" in inputs:
                feeds["start_pos"] = np.array([0], dtype=np.int64)
            if "mask" in inputs:
                feeds["mask"] = np.empty((0,), dtype=np.float32)

            # Prefill starts with 0-length past KV cache
            # Shape: (batch_size, num_kv_heads, 0, head_dim)
            for i in range(num_layers):
                feeds[f"past_key_{i}"] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)
                feeds[f"past_value_{i}"] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)

            outputs = sess.run(None, feeds)
            logits = outputs[0]
            current_kvs = outputs[1:]

            # Extract the first next token
            next_token_logits = logits[0, -1, :]
            next_token = int(np.argmax(next_token_logits))

            # Initialize state tracking for stream
            generated_tokens = [next_token]
            current_pos = prompt_len
            is_cot = False
            printed_len = 0

            # Handling of the First Generated Token
            if next_token == cot_start_id:
                console.print("[bold blue]<thinking>[/bold blue]", end="")
                is_cot = True
            elif next_token == cot_end_id:
                console.print("[bold green]答：[/bold green]", end="")
                is_cot = False
            elif next_token == im_end_id:
                console.print("\n[dim](Generation ended immediately)[/dim]")
                messages.append({'role': 'assistant', 'content': ''})
                continue
            else:
                text = tokenizer.decode([next_token])
                console.print(text, end="")
                printed_len = len(text)

            # ====================================================
            # B. Decode Stage (Streaming Generation Loop using same Unified Graph)
            # ====================================================
            max_new_tokens = 1024
            for step in range(max_new_tokens - 1):
                curr_input = np.array([[generated_tokens[-1]]], dtype=np.int64)
                
                feeds_decode = {}
                if "input_ids" in inputs:
                    feeds_decode["input_ids"] = curr_input
                if "start_pos" in inputs:
                    feeds_decode["start_pos"] = np.array([current_pos], dtype=np.int64)
                if "mask" in inputs:
                    feeds_decode["mask"] = np.empty((0,), dtype=np.float32)
                
                for i in range(num_layers):
                    feeds_decode[f"past_key_{i}"] = current_kvs[2 * i]
                    feeds_decode[f"past_value_{i}"] = current_kvs[2 * i + 1]

                outputs_decode = sess.run(None, feeds_decode)
                logits_decode = outputs_decode[0]
                current_kvs = outputs_decode[1:]

                next_token = int(np.argmax(logits_decode[0, -1, :]))
                
                # Check for termination/special tokens before rendering
                if next_token == im_end_id:
                    if is_cot:
                        console.print("[bold blue]</thinking>[/bold blue]")
                    break

                if next_token == cot_start_id:
                    console.print("[bold blue]<thinking>[/bold blue]", end="")
                    is_cot = True
                    generated_tokens.append(next_token)
                    current_pos += 1
                    continue
                elif next_token == cot_end_id:
                    console.print("\n\n[bold green]答：[/bold green]", end="")
                    is_cot = False
                    generated_tokens.append(next_token)
                    current_pos += 1
                    continue

                generated_tokens.append(next_token)
                current_pos += 1

                # Decode only the active generated content to support clean stream printing
                clean_tokens = [t for t in generated_tokens if t not in (cot_start_id, cot_end_id, im_end_id, pad_id)]
                full_text = tokenizer.decode(clean_tokens)
                new_text = full_text[printed_len:]
                
                if new_text:
                    if is_cot:
                        console.print(new_text, end="", style="blue")
                    else:
                        console.print(new_text, end="")
                    printed_len = len(full_text)

            console.print() # Newline at end of turn
            
            # Store generated assistant response in messages to maintain history
            clean_resp_tokens = [t for t in generated_tokens if t not in (cot_start_id, cot_end_id, im_end_id, pad_id)]
            assistant_text = tokenizer.decode(clean_resp_tokens)
            messages.append({'role': 'assistant', 'content': assistant_text})

        except KeyboardInterrupt:
            console.print("\n[bold red]Generation interrupted by user.[/bold red]")
            break

if __name__ == "__main__":
    main()
