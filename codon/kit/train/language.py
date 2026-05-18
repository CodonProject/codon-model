import torch


def run_sanity_check(model_instance, tokenizer, device, eos_id, step, prompt_text='The'):
    print(f'\n[*] Interface (Step {step})...')
    model_instance.eval()
    with torch.no_grad():
        try:
            if hasattr(tokenizer.fast_tokenizer, 'encode'):
                prompt_ids = tokenizer.fast_tokenizer.encode(prompt_text)
                if not isinstance(prompt_ids, list):
                    prompt_ids = prompt_ids.ids 
            else:
                prompt_ids = [0, 1, 2][:3]
                
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            
            generated_ids = model_instance.generate(
                prompt_tensor,
                max_new_tokens=50,
                temperature=0.8,
                eos_token_id=eos_id
            )
            generated_text = tokenizer.fast_tokenizer.decode(
                generated_ids[0].cpu().numpy().tolist()
            )
            print(f'[*] Output: {generated_text}\n')
        except Exception as e:
            print(f'[!] Failed: {e}\n')
        finally:
            model_instance.train()