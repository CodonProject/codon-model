import torch
import numpy as np
import onnxruntime as ort
from codon.motif import MotifA1, MotifA1Tokenizer

def verify():
    # 1. Load PyTorch model
    print("Loading PyTorch model...")
    model_pt = MotifA1().from_remote().eval()
    tokenizer = MotifA1Tokenizer().from_remote()
    
    # 2. Prepare inputs
    messages = [{"role": "user", "content": "你好"}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_len = len(prompt_ids)
    print(f"Prompt: {prompt_ids}, length: {prompt_len}")
    
    # Convert to PyTorch tensors
    input_ids_pt = torch.tensor([prompt_ids], dtype=torch.long)
    start_pos_pt = torch.tensor([0], dtype=torch.long)
    
    # Run PyTorch Prefill
    with torch.no_grad():
        outputs_pt = model_pt(
            input_ids=input_ids_pt,
            start_pos=start_pos_pt,
            use_cache=True
        )
    logits_pt = outputs_pt.logits.numpy()
    pkv_pt = outputs_pt.past_key_values
    
    next_token_pt = int(np.argmax(logits_pt[0, -1, :]))
    print(f"\n[PyTorch Prefill] Next token: {next_token_pt} ({tokenizer.decode([next_token_pt])!r})")
    print(f"[PyTorch Prefill] Logits shape: {logits_pt.shape}")
    print(f"[PyTorch Prefill] Sample logits (last token): {logits_pt[0, -1, :5]}")
    
    # 3. Load ONNX model
    print("\nLoading ONNX model...")
    sess = ort.InferenceSession("motifa1.onnx", providers=['CPUExecutionProvider'])
    
    # Run ONNX Prefill
    inputs = [node.name for node in sess.get_inputs()]
    num_layers = len([n for n in inputs if n.startswith("past_key_")])
    
    feeds = {}
    feeds["input_ids"] = np.array([prompt_ids], dtype=np.int64)
    feeds["start_pos"] = np.array([0], dtype=np.int64)
    if "mask" in inputs:
        feeds["mask"] = np.empty((0,), dtype=np.float32)
    for i in range(num_layers):
        feeds[f"past_key_{i}"] = np.zeros((1, 2, 0, 96), dtype=np.float32)
        feeds[f"past_value_{i}"] = np.zeros((1, 2, 0, 96), dtype=np.float32)
        
    outputs_onnx = sess.run(None, feeds)
    logits_onnx = outputs_onnx[0]
    pkv_onnx = outputs_onnx[1:]
    
    next_token_onnx = int(np.argmax(logits_onnx[0, -1, :]))
    print(f"[ONNX Prefill] Next token: {next_token_onnx} ({tokenizer.decode([next_token_onnx])!r})")
    print(f"[ONNX Prefill] Logits shape: {logits_onnx.shape}")
    print(f"[ONNX Prefill] Sample logits (last token): {logits_onnx[0, -1, :5]}")
    
    # Compare prefill logits
    diff_prefill = np.max(np.abs(logits_pt - logits_onnx))
    print(f"Max absolute prefill logits difference: {diff_prefill}")
    
    # Compare KV Cache after Prefill
    print("\n--- KV Cache Comparison (Layer 0 Key) ---")
    pkv_pt_layer0_k = pkv_pt[0][0].numpy()
    pkv_onnx_layer0_k = pkv_onnx[0]
    print(f"PyTorch Layer 0 Key Cache Shape: {pkv_pt_layer0_k.shape}")
    print(f"ONNX Layer 0 Key Cache Shape: {pkv_onnx_layer0_k.shape}")
    print(f"PyTorch Layer 0 Key sample values (first 5 of head 0, pos 0): {pkv_pt_layer0_k[0, 0, 0, :5]}")
    print(f"ONNX Layer 0 Key sample values (first 5 of head 0, pos 0): {pkv_onnx_layer0_k[0, 0, 0, :5]}")
    diff_pkv = np.max(np.abs(pkv_pt_layer0_k - pkv_onnx_layer0_k))
    print(f"Max absolute Layer 0 Key Cache difference: {diff_pkv}")
    
    # 4. Decode Step 1
    # PyTorch Decode Step 1
    input_ids_pt_dec = torch.tensor([[next_token_pt]], dtype=torch.long)
    start_pos_pt_dec = torch.tensor([prompt_len], dtype=torch.long)
    
    with torch.no_grad():
        outputs_pt_dec = model_pt(
            input_ids=input_ids_pt_dec,
            start_pos=start_pos_pt_dec,
            past_key_values=pkv_pt,
            use_cache=True
        )
    logits_pt_dec = outputs_pt_dec.logits.numpy()
    pkv_pt_dec = outputs_pt_dec.past_key_values
    
    next_token_pt_dec = int(np.argmax(logits_pt_dec[0, -1, :]))
    print(f"\n[PyTorch Decode Step 1] Next token: {next_token_pt_dec} ({tokenizer.decode([next_token_pt_dec])!r})")
    print(f"[PyTorch Decode Step 1] Sample logits: {logits_pt_dec[0, -1, :5]}")
    
    # ONNX Decode Step 1
    feeds_dec = {}
    feeds_dec["input_ids"] = np.array([[next_token_onnx]], dtype=np.int64)
    feeds_dec["start_pos"] = np.array([prompt_len], dtype=np.int64)
    if "mask" in inputs:
        feeds_dec["mask"] = np.empty((0,), dtype=np.float32)
    for i in range(num_layers):
        feeds_dec[f"past_key_{i}"] = pkv_onnx[2 * i]
        feeds_dec[f"past_value_{i}"] = pkv_onnx[2 * i + 1]
        
    outputs_onnx_dec = sess.run(None, feeds_dec)
    logits_onnx_dec = outputs_onnx_dec[0]
    pkv_onnx_dec = outputs_onnx_dec[1:]
    
    next_token_onnx_dec = int(np.argmax(logits_onnx_dec[0, -1, :]))
    print(f"[ONNX Decode Step 1] Next token: {next_token_onnx_dec} ({tokenizer.decode([next_token_onnx_dec])!r})")
    print(f"[ONNX Decode Step 1] Sample logits: {logits_onnx_dec[0, -1, :5]}")
    
    # Compare decode step 1 logits
    diff_dec = np.max(np.abs(logits_pt_dec - logits_onnx_dec))
    print(f"Max absolute decode step 1 logits difference: {diff_dec}")

if __name__ == "__main__":
    verify()
