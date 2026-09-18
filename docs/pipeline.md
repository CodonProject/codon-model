# Training Pipelines Documentation

多模态（图片 / 音频）接入 `codon.pipeline.pretrain` 与 `codon.pipeline.sft` 的约定。
占位符语义见 [Session](utils/session.md)，模型侧契约见 `codon/motif/chord/model.py`
与 [Media](utils/media.md)。

---

## Pretrain

```python
@configclass
class PretrainConfig:
    compiled: bool = True
    batch_builder: Optional[Callable] = None     # batch -> (model_kwargs, labels)
    ...
```

`train_step` 的流程：

```python
model_kwargs, labels = self._unpack_batch(batch)          # 默认: ({'input_ids': inputs}, labels)
model_kwargs = {k: move_to_device(v, self.device) for k, v in model_kwargs.items()}
output = self.model(**model_kwargs)
loss = F.cross_entropy(output.logits.view(-1, V), labels.view(-1))   # ignore_index=-100
```

* `batch_builder` 是**显式契约**：想喂什么就返回什么，训练循环原样 `**` 展开；
* `move_to_device` 递归搬张量，保留「每 batch 行一个子列表」的嵌套结构；
* 模态位置在 `labels` 里写 `-100`（`Session.to_tensors()` 已经把占位符 mask 掉）。

多模态 batch 示例：

```python
cfg = PretrainConfig(
    compiled=False,                                     # 见文末注意事项
    batch_builder=lambda b: (
        {'input_ids': b['input_ids'],                   # 含 <|modality_image_pad|>
         'images': b['images'],                         # 扁平 list 或 [[img], [img]]
         'image_patch_indices': b['image_patch_indices'],  # 也可以只给 'image_patch_id'
         'mask': b['attention_mask'],
         'audios': b['audios'],
         'audio_patch_indices': b['audio_patch_indices']},
        b['labels'],
    ),
)
pipe = PretrainPipeline(model, tokenizer=tokenizer, config=cfg)
```

数据侧：`ChunkedTokenStream` 只产出纯文本 `(inputs, labels)`，多模态需要自带 stream
（参考 `demo_vlm_pretrain.py` 的 `ToyVLMStream` + `VLMPretrainPipeline` 写法）。

---

## SFT

`SFTPipeline.train_step` 按 **forward 签名**从 batch dict 里挑字段：

| batch 字段 | forward 形参 | 说明 |
|-----------|-------------|------|
| `input_ids` | `input_ids` | 必填 |
| `images` / `image_patch_indices` / `image_patch_id` | 同名 | 多模态占位符注入 |
| `audios` / `audio_patch_indices` / `audio_patch_id` | 同名 | 同上（mel） |
| `attention_mask` | `mask` | 0/1 右填充掩码，[B, L] 会被展开成 [B, 1, 1, L] |

纯文本模型（`forward` 不收这些形参）只会拿到 `input_ids`，不会被多模态字段弄出 `TypeError`
（`codon.pipeline.base.select_model_kwargs`）。

**占位符参数按模型推导**：`SFTPipeline` 在构建 stage 时读取
`image_patch_size` / `audio_pool_stride`（`MotifChord` = 16 / 8）并注入数据集；
`CodonSFT` / `MotifSFT` 都接受 `patch_size` / `audio_pool_stride`（自定义数据集没有这两个
形参时会自动跳过）。也可以显式覆盖：

```python
cfg = SFTConfig(
    stage_specs=[{'name': 's0', 'folder': './data', 'epochs': 1, 'ckpt': './s0'}],
    dataset_kwargs={'patch_size': 16, 'audio_pool_stride': 8},   # 显式值优先
    lora=None,                                                   # 见文末注意事项
)
pipe = SFTPipeline(model, tokenizer=tokenizer, config=cfg)
```

### 关于 padding

SFT 数据集把 batch 右填充到 `pad_length`，这对 chord 是安全的：因果注意力与线性（GDN）
递归都只向后传播，padding 在末尾不会影响真实 token 的位置；同时 `labels` 在 padding 处是
`-100`，GQA 层的 padding key 也会被 `mask` 屏蔽（GDN 层本来就不消费 `attention_mask`）。
左填充或 pack 序列请自行保证语义。

---

## 注意事项

1. **LoRA 与冻结塔**：`lora=LoRAConfig(target='all-linear')` 会把冻结的视觉 / 音频塔里的
   `nn.Linear` 一起包上 LoRA（`freeze_backbone` 随后只放开 `lora_` / `dora_` 参数），
   等于顺带适配编码器。不想要这个行为就显式排除：
   `LoRAConfig(target='all-linear', module_exclude=['vision', 'audio'])`。
2. **`compiled=True`**：两个模态塔已经用 `@torch.compiler.disable` 留在图外；占位符注入
   本身是数据相关的 in-place scatter，`dynamic=True` 下按 batch 形状反复编译不划算，
   多模态训练建议 `compiled=False`（与 `demo_vlm_pretrain.py` 一致）。
3. **多模态 SFT 数据**：`CodonSFT` / `MotifSFT` 目前只把文本 turn（`input` / `content` /
   `cot`）拼进 `Session`，不会解析媒体列；要真正喂图，需要数据集把媒体项写成
   `{'type': 'image', 'image': tensor}` / `{'type': 'audio', 'audio': mel}` 的内容列表
   （`Session` 与 `normalize_messages` 已经支持这两种形态）。
4. **模态位置不算 loss**：展开出来的占位符 `ignore_mask=True`，`labels` 为 `-100`，
   不需要额外处理。
