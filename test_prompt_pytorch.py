import torch
from codon.motif import MotifA1, MotifA1Tokenizer

def test_pytorch():
    print("Loading PyTorch model...")
    model = MotifA1().from_remote().eval()
    tokenizer = MotifA1Tokenizer().from_remote()
    
    messages = [{"role": "user", "content": "你好"}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    
    print(f"Prompt IDs: {prompt_ids}")
    print("Generating...")
    
    # Generate using the built-in generate method (using KV cache by default)
    output_ids = model.generate(prompt_tensor, max_new_tokens=20, eos_token_id=None, use_cache=True)
    output_list = output_ids[0].tolist()
    
    print(f"Generated IDs: {output_list}")
    print(f"Decoded: {tokenizer.decode(output_list)}")

if __name__ == "__main__":
    test_pytorch()
