import numpy as np
import onnxruntime as ort
from codon.motif import MotifA1Tokenizer


def _np_float_dtype(type_str: str):
    '''
    Maps an ORT input type string to the matching numpy floating-point dtype.

    Args:
        type_str (str): The type string from `ort.NodeArg.type`, e.g. 'tensor(float)'
            or 'tensor(float16)'.

    Returns:
        numpy.dtype: `np.float16` for FP16 graphs, `np.float32` otherwise.
    '''
    return np.float16 if 'float16' in type_str else np.float32


def test_prompt():
    tokenizer = MotifA1Tokenizer().from_remote()
    sess = ort.InferenceSession("motifa1.onnx", providers=['CPUExecutionProvider'])

    # Simple prompt
    messages = [{"role": "user", "content": "你好"}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_len = len(prompt_ids)
    print(f"Prompt IDs: {prompt_ids}")
    print(f"Prompt length: {prompt_len}")

    input_meta = {node.name: node for node in sess.get_inputs()}
    inputs = list(input_meta.keys())
    num_layers = len([n for n in inputs if n.startswith("past_key_")])

    # Auto-detect float dtype expected by the graph (FP32 vs FP16 export)
    float_dtype = _np_float_dtype(input_meta['mask'].type) if 'mask' in input_meta else np.float32
    print(f"Detected float dtype: {float_dtype.__name__}")

    feeds = {}
    if "input_ids" in inputs:
        feeds["input_ids"] = np.array([prompt_ids], dtype=np.int64)
    if "start_pos" in inputs:
        feeds["start_pos"] = np.array([0], dtype=np.int64)
    if "mask" in inputs:
        feeds["mask"] = np.empty((0,), dtype=float_dtype)

    for i in range(num_layers):
        feeds[f"past_key_{i}"] = np.zeros((1, 2, 0, 96), dtype=float_dtype)
        feeds[f"past_value_{i}"] = np.zeros((1, 2, 0, 96), dtype=float_dtype)

    outputs = sess.run(None, feeds)
    logits = outputs[0]
    current_kvs = outputs[1:]

    next_token = int(np.argmax(logits[0, -1, :]))
    print(f"Next token ID after prefill: {next_token}")
    print(f"Decoded: {tokenizer.decode([next_token])}")

    # Let's run a few decode steps
    generated = [next_token]
    current_pos = prompt_len

    for step in range(15):
        curr_input = np.array([[generated[-1]]], dtype=np.int64)
        feeds_decode = {}
        if "input_ids" in inputs:
            feeds_decode["input_ids"] = curr_input
        if "start_pos" in inputs:
            feeds_decode["start_pos"] = np.array([current_pos], dtype=np.int64)
        if "mask" in inputs:
            feeds_decode["mask"] = np.empty((0,), dtype=float_dtype)

        for i in range(num_layers):
            feeds_decode[f"past_key_{i}"] = current_kvs[2 * i]
            feeds_decode[f"past_value_{i}"] = current_kvs[2 * i + 1]

        outputs_decode = sess.run(None, feeds_decode)
        logits_decode = outputs_decode[0]
        current_kvs = outputs_decode[1:]

        next_token = int(np.argmax(logits_decode[0, -1, :]))
        generated.append(next_token)
        current_pos += 1
        print(f"Step {step+1}: Token ID = {next_token}, Decoded = {tokenizer.decode([next_token])!r}")


if __name__ == '__main__':
    test_prompt()
