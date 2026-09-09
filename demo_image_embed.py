'''
如何把 <|image_start|> / <|image_end|> 之间的图像特征嵌入 test_mbv3.py 示例。

链路（框架已内置，不需要改模型）：

    1. Session 按 chat template 生成 token 序列，图像位置渲染出 [image_start][image_end]
    2. Session._expand_image_patches 把两 token 之间的空隙填成 N 个 image_patch
       N = (h // patch_size) * (w // patch_size)，与 MotifV1 的 patch 切分一致
    3. Session.to_tensors() 返回 input_ids / attention_mask / image_patch_indices / images
       image_patch_indices 就是「占位 token 在序列里的绝对位置」
    4. 模型侧 x[image_patch_indices] = vision_emb.embed_image(img)  完成嵌入

本文件用本地最小 tokenizer 跑通 1-3 与 4 的等价实现，不依赖远程权重。
'''
from codon import *
from codon.utils.tokens import PackedTokenizer
from codon.utils.session import Session

from tokenizers import Tokenizer
from tokenizers.models import WordLevel

# ---------------------------------------------------------------- 1. 最小 tokenizer
# 真实场景用 MotifA1Tokenizer().from_remote()，这里为了可跑通构造本地词表。
SPECIALS = [
    '<|unk|>', '<|im_start|>', '<|im_end|>', '<|system|>', '<|user|>', '<|model|>',
    '<|thought_start|>', '<|thought_end|>', '<|modality_image_start|>',
    '<|modality_image_end|>', '<|modality_image_pad|>', '<|pad|>',
    '[unused_42]',          # PackedTokenizer 默认转义 token，词表里必须有
]
WORDS = ['hello', 'world']
VOCAB = {tok: i for i, tok in enumerate(SPECIALS + WORDS)}

TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}"
    "{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for it in m['content'] %}"
    "{% if it['type'] == 'image' %}<|modality_image_start|><|modality_image_end|>"
    "{% else %}{{ it['text'] }}{% endif %}"
    "{% endfor %}{% endif %}"
    "<|im_end|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>model<|thought_start|>{% endif %}"
)


def build_tokenizer() -> PackedTokenizer:
    raw = Tokenizer(WordLevel(VOCAB, unk_token='<|unk|>'))
    # 关键：把特殊 token 注册进 tokenizer，否则会被当成 unk
    raw.add_special_tokens(SPECIALS)
    tok = PackedTokenizer(raw)
    tok.set_chat_template(TEMPLATE)
    tok.config['unk_token'] = '<|unk|>'
    tok.config['pad_token'] = '<|pad|>'
    tok.config['eos_token'] = '<|im_end|>'
    return tok


def main():
    patch_size = 12
    tokenizer = build_tokenizer()
    session = Session(tokenizer, patch_size=patch_size)

    # 假设已知这两个 id（Session 内部也会用 token_to_id 解析同名 token）
    image_start_id = tokenizer.token_to_id('<|modality_image_start|>')
    image_end_id = tokenizer.token_to_id('<|modality_image_end|>')
    image_patch_id = tokenizer.token_to_id('<|modality_image_pad|>')
    print(f'image_start={image_start_id}  image_end={image_end_id}  image_patch={image_patch_id}')

    # 96x48 的图 -> (96//12) * (48//12) = 8 * 4 = 32 个 patch
    image = torch.randn(3, 96, 48)
    session.add_message({
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': 'hello world'},
        ],
    })

    tensors = session.to_tensors(device='cpu', batch_dim=True)
    input_ids = tensors['input_ids']                 # [1, seq_len]
    patch_idx = tensors['image_patch_indices']       # [1, num_patches]
    print('input_ids      :', input_ids.shape)
    print('patch indices  :', patch_idx.shape, patch_idx.tolist())
    print('images         :', [tuple(i.shape) for i in tensors['images']])
    print('decoded        :', tokenizer.decode(input_ids[0].tolist()))

    # 校验：start/end 之间确实被填成了 32 个 patch 占位符
    ids = input_ids[0].tolist()
    s = ids.index(image_start_id)
    e = ids.index(image_end_id)
    between = ids[s + 1:e]
    assert len(between) == 32, len(between)
    assert all(t == image_patch_id for t in between)
    assert patch_idx[0].tolist() == list(range(s + 1, e))

    # ------------------------------------------------------------ 2. 嵌入图像特征
    # 模型内部就是这一句：x[b, indices] = 图像特征
    model_dim = 64
    token_emb = nn.Embedding(tokenizer.vocab_size, model_dim)
    x = token_emb(input_ids)                                   # [1, seq_len, model_dim]

    num_patches = 32
    img_features = torch.randn(num_patches, model_dim)          # 来自视觉编码器

    b = 0
    idx_b = patch_idx[b]
    idx_b = idx_b[idx_b >= 0]
    n = min(len(idx_b), img_features.size(0))
    x[b, idx_b[:n]] = img_features[:n]

    # 确认占位位置已被替换、其它位置未动
    assert torch.allclose(x[b, idx_b[:n]], img_features[:n])
    print('嵌入后 x:', tuple(x.shape), '-> 占位符位置已写入视觉特征')

    # ------------------------------------------------------------ 3. 用真实模型（可选）
    # 需要远程权重：MotifA1.from_remote() / MotifV1.from_remote()
    #
    #   from codon.motif import MotifA1
    #   from codon.utils.session import Session
    #   from vl import MotifA1_VL
    #
    #   vl = MotifA1_VL()                      # 内部含 MotifA1 + MotifV1 + VisionEmbedding
    #   tok = MotifA1Tokenizer().from_remote()
    #   session = Session(tok, patch_size=12)  # patch_size 必须与 MotifV1 的 patch_size 一致
    #   session.add_message({'role': 'user', 'content': [
    #       {'type': 'image', 'image': image},
    #       {'type': 'text', 'text': '描述这张图'},
    #   ]})
    #   t = session.to_tensors(device='cuda', batch_dim=True)
    #   out = vl(
    #       input_ids=t['input_ids'],
    #       images=t['images'],
    #       image_patch_indices=t['image_patch_indices'],
    #       mask=t['attention_mask'],
    #   )
    #
    # MotifA1_VL.forward 内部对每个 batch：
    #   emb = self.vision_emb.embed_image(img.unsqueeze(0))     # [1, patches, hidden_dim]
    #   x[b, b_indices[:n]] = emb.flatten(0, 1)[:n]

    print('OK')


if __name__ == '__main__':
    main()
