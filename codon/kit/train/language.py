import torch

from codon.utils.session import Session
from codon.utils.tokens  import PackedTokenizer

from tqdm import tqdm

def run_sanity_check(model_instance, tokenizer, device, eos_id, step, prompt_text='The') -> str:
    tqdm.write(f'\n[*] Interface (Step {step})...')
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
                eos_token_id=eos_id,
            )
            generated_text = tokenizer.fast_tokenizer.decode(
                generated_ids[0].cpu().numpy().tolist()
            )
            tqdm.write(f'[*] Output: {generated_text}\n')
        except Exception as e:
            tqdm.write(f'[!] Failed: {e}\n')
        finally:
            model_instance.train()

        return generated_text


def run_chat_turn(
    model_instance,
    tokenizer: PackedTokenizer,
    device,
    step: int,
    user_prompt: str = 'Hello.',
    system_prompt: str = 'You are a helpful assistant.',
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = None,
    enable_thinking: bool = False,
    eos_token: str = '[im_end]',
    print_full: bool = False,
):
    '''
    Run a single-turn chat against the model using the Session / chat-template stack.

    Builds: [system] system_prompt -> [user] user_prompt -> [model] (generation prompt).
    Generates up to `max_new_tokens` and decodes only the assistant's response.

    Args:
        model_instance (CausalLanguageModel): The LM. Must implement `.generate(...)`.
        tokenizer (PackedTokenizer): The packed tokenizer with chat template.
        device (torch.device | str): Target device for the prompt tensor.
        step (int): Current training step, for log prefix.
        user_prompt (str): User content of this turn.
        system_prompt (str): System persona/instruction.
        max_new_tokens (int): Generation budget.
        temperature (float): Sampling temperature.
        top_k (int, optional): If set, top-k truncation.
        enable_thinking (bool): Whether to open a [cot_start] section before content.
        eos_token (str): Special token whose id is used as the eos signal.
        print_full (bool): If True, also print the full decoded sequence (incl. prompt).
    '''
    print(f'\n[*] Chat turn (Step {step})...')
    model_instance.eval()

    was_training = model_instance.training
    try:
        # Build the prompt via Session so the chat template is faithfully reproduced.
        session = Session(tokenizer)
        session.add_message({'role': 'system', 'content': system_prompt})
        session.add_message({'role': 'user',   'content': user_prompt})
        session.add_generation_prompt(enable_thinking=enable_thinking)

        prompt_ids = session.input_ids
        prompt_len = len(prompt_ids)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        eos_id = tokenizer.token_to_id(eos_token)
        if eos_id is None:   # A2 尖括号词表：把方括号默认名映射到 <|...|> 等价 token
            alt = {
                '[im_end]': '<|im_end|>', '[im_start]': '<|im_start|>',
                '[cot_start]': '<|thought_start|>', '[cot_end]': '<|thought_end|>',
                '[pad]': '<|pad|>',
            }.get(eos_token)
            eos_id = tokenizer.token_to_id(alt) if alt else None

        sampler = None
        if top_k is not None:   # CausalLanguageModel.generate 无 top_k 参数，改用 sampler
            from codon.model.sampler import Sampler
            sampler = Sampler(temperature=temperature, top_k=top_k)
        with torch.no_grad():
            generated = model_instance.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                sampler=sampler,
                eos_token_id=eos_id,
            )

        full_ids = generated[0].detach().cpu().tolist()
        new_ids  = full_ids[prompt_len:]

        # Strip a trailing eos for cleaner display, but keep it in raw output.
        display_ids = new_ids[:-1] if (new_ids and new_ids[-1] == eos_id) else new_ids

        reply = tokenizer.decode(display_ids, skip_special_tokens=False)

        print(f'[*] User    : {user_prompt}')
        print(f'[*] Assistant: {reply}')
        if print_full:
            full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
            print(f'[*] Full    : {full_text}')
        print(f'[*] Stats   : prompt={prompt_len} tok, generated={len(new_ids)} tok\n')

        return {
            'prompt_ids':    prompt_ids,
            'generated_ids': new_ids,
            'reply':         reply,
        }

    except Exception as e:
        print(f'[!] Failed: {e}\n')
        return None
    finally:
        if was_training:
            model_instance.train()