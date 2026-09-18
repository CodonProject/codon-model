'''
多模态链路接入 utils/generate + utils/service 的测试。

用一个「假多模态 LM」而不是真实 MotifChord（后者 __init__ 要拉远程权重），
假模型把每次 forward 的模态参数记下来，于是可以断言：
    1. 图片 / 音频载荷（data URL、mel 张量、npy）被解码成正确形状的张量；
    2. chat() 只在 prefill 注入模态，decode 步不带模态，且 mask 一并传下去；
    3. 模型没声明模态能力时抛 UnsupportedModalityError / 返回 400；
    4. 服务端 400 分类正确（unsupported_modality vs invalid_content）。
'''

import asyncio
import base64
import io
import json
import os
import sys
import wave

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import unittest
from unittest import mock

from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from codon.res import LM
from codon.model.types.language import CausalLanguageModel, CausalLanguageModelOutput
from codon.utils.generate import chat
from codon.utils.media import (
    UnsupportedModalityError,
    has_media,
    load_image,
    load_mel,
    modal_config,
    model_modalities,
    normalize_messages,
)
from codon.utils.service import ChatCompletionRequest, ModelCard, Service
from codon.utils.session import Session
from codon.utils.tokens import PackedTokenizer


# ---------------------------------------------------------------- 最小 tokenizer

SPECIALS = [
    '<|unk|>', '<|im_start|>', '<|im_end|>', '<|system|>', '<|user|>', '<|model|>',
    '<|thought_start|>', '<|thought_end|>',
    '<|effort_low|>', '<|effort_high|>', '<|effort_max|>',
    '<|modality_image_start|>', '<|modality_image_pad|>', '<|modality_image_end|>',
    '<|modality_audio_start|>', '<|modality_audio_pad|>', '<|modality_audio_end|>',
    '<|pad|>', '[unused_42]',
]
WORDS = ['hello', 'world', 'hi']


def build_tokenizer() -> PackedTokenizer:
    vocab = {tok: i for i, tok in enumerate(SPECIALS + WORDS)}
    raw = Tokenizer(WordLevel(vocab, unk_token='<|unk|>'))
    raw.add_special_tokens(SPECIALS)

    tok = PackedTokenizer(raw)
    with open(LM['jinja'], encoding='utf-8') as f:
        tok.set_chat_template(f.read())
    tok.config['unk_token'] = '<|unk|>'
    tok.config['pad_token'] = '<|pad|>'
    tok.config['eos_token'] = '<|im_end|>'
    return tok


# ---------------------------------------------------------------- 假模型

class _FakeMultimodalLM(CausalLanguageModel):
    '''只做嵌入 + 线性头，并把每次 forward 的模态参数记录下来。'''

    supports_image = True
    supports_audio = True
    image_patch_size = 16
    audio_pool_stride = 8
    audio_num_mel_bins = 80

    def __init__(self, vocab_size: int, model_dim: int = 16):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, model_dim)
        self.proj_out = torch.nn.Linear(model_dim, vocab_size, bias=False)
        self.calls = []

    def forward(
        self,
        input_ids,
        images=None,
        image_patch_indices=None,
        audios=None,
        audio_patch_indices=None,
        mask=None,
        start_pos=0,
        past_key_values=None,
        output_attentions=False,
    ) -> CausalLanguageModelOutput:
        self.calls.append({
            'seq_len': int(input_ids.shape[1]),
            'images': images,
            'image_patch_indices': image_patch_indices,
            'audios': audios,
            'audio_patch_indices': audio_patch_indices,
            'mask': mask,
            'start_pos': start_pos,
        })
        x = self.token_emb(input_ids)
        if images is not None:
            for b in range(x.shape[0]):
                idx = image_patch_indices[b]
                idx = idx[idx >= 0]
                x[b, idx] = 0.5                     # 标记图像占位符
        if audios is not None:
            for b in range(x.shape[0]):
                idx = audio_patch_indices[b]
                idx = idx[idx >= 0]
                x[b, idx] = -0.5                    # 标记音频占位符
        return CausalLanguageModelOutput(logits=self.proj_out(x))


class _TextOnlyLM(CausalLanguageModel):
    '''未声明任何模态能力。'''

    def __init__(self, vocab_size: int, model_dim: int = 16):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, model_dim)
        self.proj_out = torch.nn.Linear(model_dim, vocab_size, bias=False)
        self.calls = []

    def forward(self, input_ids, start_pos=0, past_key_values=None, output_attentions=False):
        self.calls.append({'seq_len': int(input_ids.shape[1])})
        return CausalLanguageModelOutput(logits=self.proj_out(self.token_emb(input_ids)))


# ---------------------------------------------------------------- 媒体载荷构造

def png_bytes(width: int = 32, height: int = 32, color=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new('RGB', (width, height), color).save(buffer, format='PNG')
    return buffer.getvalue()


def png_data_url(width: int = 32, height: int = 32) -> str:
    return 'data:image/png;base64,' + base64.b64encode(png_bytes(width, height)).decode()


def wav_bytes(seconds: float = 0.4, sample_rate: int = 16000, freq: float = 440.0) -> bytes:
    frames = int(seconds * sample_rate)
    samples = (np.sin(2 * np.pi * freq * np.arange(frames) / sample_rate) * 8000).astype('<i2')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


class TestMediaDecoding(unittest.TestCase):
    '''codon/utils/media.py：载荷 -> 张量'''

    def test_load_image_forms_agree(self):
        expected = torch.full((3, 32, 32), 1.0)      # 纯红
        expected[1:] = 0.0

        forms = {
            'tensor': expected.clone(),
            'numpy_chw': expected.numpy().copy(),
            'numpy_hwc': expected.permute(1, 2, 0).numpy().copy(),
            'pil': Image.fromarray(
                (expected.permute(1, 2, 0).numpy() * 255).astype('uint8')
            ),
            'bytes': png_bytes(),
            'data_url': png_data_url(),
            'base64': base64.b64encode(png_bytes()).decode(),
            'dict': {'url': png_data_url()},
        }

        for name, source in forms.items():
            image = load_image(source)
            self.assertEqual(image.shape, (3, 32, 32), name)
            self.assertTrue(torch.allclose(image, expected, atol=1e-2), name)

    def test_load_image_from_path(self):
        path = os.path.join(project_root, '_tmp_media_test.png')
        with open(path, 'wb') as f:
            f.write(png_bytes(48, 32))
        try:
            image = load_image(path)
        finally:
            os.remove(path)
        self.assertEqual(image.shape, (3, 32, 48))

    def test_load_image_rejects_unknown_payload(self):
        with self.assertRaises((ValueError, TypeError)):
            load_image(12345)

    def test_load_mel_tensor_and_npy(self):
        mel = torch.randn(80, 64)
        self.assertTrue(torch.equal(load_mel(mel), mel))
        self.assertTrue(torch.equal(load_mel(mel.numpy()), mel))

        buffer = io.BytesIO()
        np.save(buffer, mel.numpy())
        self.assertTrue(torch.allclose(load_mel(buffer.getvalue()), mel))
        self.assertTrue(torch.allclose(
            load_mel('data:application/octet-stream;base64,' + base64.b64encode(buffer.getvalue()).decode()),
            mel,
        ))

    def test_load_mel_transposes_channel_last(self):
        mel = torch.randn(64, 80)                   # frames first
        self.assertTrue(torch.equal(load_mel(mel), mel.transpose(0, 1)))

    def test_load_mel_rejects_wrong_bins(self):
        with self.assertRaises(ValueError):
            load_mel(torch.randn(64, 64))

    def test_load_mel_from_waveform(self):
        try:
            import torchaudio  # noqa: F401
        except ImportError:                          # pragma: no cover - 取决于环境
            self.skipTest('torchaudio not installed')

        mel = load_mel(wav_bytes(seconds=1.0))
        self.assertEqual(mel.shape[0], 80)
        # 1s / hop 160 -> 100 帧左右（centered STFT 允许 ±1 帧）
        self.assertLess(abs(mel.shape[1] - 100), 3)

        # 与 Session 的占位符展开口径一致
        session = Session(build_tokenizer(), audio_pool_stride=8)
        expected = ((mel.shape[1] - 1) // 2 + 1) // 8
        self.assertEqual(session._audio_token_count(mel), max(1, expected))

    def test_load_mel_truncates_to_whisper_window(self):
        mel = torch.randn(80, 5000)
        self.assertEqual(load_mel(mel).shape[1], 3000)

    def test_has_media_and_normalize(self):
        buffer = io.BytesIO()
        np.save(buffer, torch.randn(80, 64).numpy())
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'hello'},
            {'type': 'image_url', 'image_url': {'url': png_data_url(32, 48)}},
            {'type': 'input_audio', 'input_audio': {
                'data': base64.b64encode(buffer.getvalue()).decode(),
                'format': 'npy',
            }},
        ]}]

        self.assertEqual(has_media(messages), (True, True, False))

        normalized = normalize_messages(messages, image_capab=True, audio_capab=True)
        items = normalized[0]['content']
        self.assertEqual(items[1]['type'], 'image')
        self.assertEqual(items[1]['image'].shape, (3, 48, 32))
        self.assertEqual(items[2]['type'], 'audio')
        self.assertEqual(items[2]['audio'].shape, (80, 64))

    def test_normalize_rejects_unsupported_modality(self):
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': png_data_url()},
        ]}]
        with self.assertRaises(UnsupportedModalityError):
            normalize_messages(messages, image_capab=False)

        video = [{'role': 'user', 'content': [
            {'type': 'video', 'video': 'whatever.mp4'},
        ]}]
        with self.assertRaises(UnsupportedModalityError):
            normalize_messages(video, video_capab=False)

    def test_normalize_is_idempotent_for_tensors(self):
        image = torch.rand(3, 32, 32)
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}]}]
        normalized = normalize_messages(messages, image_capab=True)
        self.assertTrue(torch.equal(normalized[0]['content'][0]['image'], image))


class TestModelModalConfig(unittest.TestCase):
    '''模型侧契约：patch 尺寸 / 池化步长 / mel 维数'''

    def test_text_model_falls_back_to_defaults(self):
        model = _TextOnlyLM(32)
        self.assertIsNone(model.image_patch_size)
        self.assertEqual(modal_config(model), (12, 8, 80))
        self.assertEqual(model_modalities(model), {
            'image': False, 'audio': False, 'video': False, 'thinking': False, 'tool': False,
        })

    def test_motifchord_reads_config(self):
        from codon.motif.chord.config import MotifChordConfig
        from codon.motif.chord.model import MotifChord

        model = MotifChord.__new__(MotifChord)
        torch.nn.Module.__init__(model)
        model.config = MotifChordConfig(
            vision_patch_size=16, audio_pool_stride=8, audio_num_mel_bins=80,
        )

        self.assertEqual(modal_config(model), (16, 8, 80))
        self.assertEqual(model_modalities(model), {
            'image': True, 'audio': True, 'video': False, 'thinking': True, 'tool': False,
        })


class TestChatMultimodal(unittest.TestCase):
    '''codon/utils/generate.py：chat() 的多模态通路'''

    def setUp(self):
        self.tokenizer = build_tokenizer()
        self.model = _FakeMultimodalLM(self.tokenizer.vocab_size)

    def _messages(self):
        return [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'hello'},
            {'type': 'image', 'image': png_data_url(32, 32)},        # 32/16 -> 4 patch
            {'type': 'audio', 'audio': torch.randn(80, 64)},         # ((64-1)//2+1)//8 = 4
        ]}]

    def _run(self, messages, **kwargs):
        return list(chat(
            model=self.model,
            tokenizer=self.tokenizer,
            device=torch.device('cpu'),
            messages=messages,
            max_new_tokens=3,
            enable_thinking=False,
            **kwargs,
        ))

    def test_chat_injects_modality_at_prefill_only(self):
        chunks = self._run(self._messages())

        prefill = self.model.calls[0]
        self.assertEqual(len(prefill['images']), 1)
        self.assertEqual(tuple(prefill['images'][0].shape), (3, 32, 32))
        self.assertEqual(len(prefill['audios']), 1)
        self.assertEqual(tuple(prefill['audios'][0].shape), (80, 64))

        # 32/16 -> 4 个图像占位符、((64-1)//2+1)//8 = 4 个音频占位符，位置连续
        image_positions = prefill['image_patch_indices'][0].tolist()
        audio_positions = prefill['audio_patch_indices'][0].tolist()
        self.assertEqual(len(image_positions), 4)
        self.assertEqual(len(audio_positions), 4)
        self.assertEqual([b - a for a, b in zip(image_positions, image_positions[1:])], [1, 1, 1])
        self.assertEqual([b - a for a, b in zip(audio_positions, audio_positions[1:])], [1, 1, 1])
        self.assertIsNotNone(prefill['mask'])
        self.assertEqual(int(prefill['mask'].sum()), prefill['seq_len'])

        # decode 步：只喂 1 个 token，不带模态
        self.assertGreaterEqual(len(self.model.calls), 2)
        for call in self.model.calls[1:]:
            self.assertEqual(call['seq_len'], 1)
            self.assertIsNone(call['images'])
            self.assertIsNone(call['audios'])

        # 假模型是随机初始化的，可能提前采到 im_end / pad -> 'stop'，两种结束原因都合法
        self.assertIn(chunks[-1].finish_reason, ('stop', 'length'))
        self.assertTrue(all(chunk.finish_reason is None for chunk in chunks[:-1]))

    def test_chat_accepts_path_and_pil_image(self):
        path = os.path.join(project_root, '_tmp_chat_image.png')
        with open(path, 'wb') as f:
            f.write(png_bytes(32, 32))
        try:
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': path},
            ]}]
            self._run(messages)
        finally:
            os.remove(path)

        prefill = self.model.calls[0]
        self.assertEqual(tuple(prefill['images'][0].shape), (3, 32, 32))
        self.assertEqual(prefill['image_patch_indices'].shape[1], 4)

    def test_chat_patch_size_override(self):
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': torch.rand(3, 32, 32)},
        ]}]
        self._run(messages, patch_size=8)                            # 32/8 -> 16 patch
        prefill = self.model.calls[0]
        self.assertEqual(prefill['image_patch_indices'].shape[1], 16)

    def test_chat_rejects_image_for_text_model(self):
        text_model = _TextOnlyLM(self.tokenizer.vocab_size)
        with self.assertRaises(UnsupportedModalityError):
            list(chat(
                model=text_model,
                tokenizer=self.tokenizer,
                device=torch.device('cpu'),
                messages=[{'role': 'user', 'content': [
                    {'type': 'image', 'image': png_data_url()},
                ]}],
                max_new_tokens=2,
            ))
        self.assertEqual(text_model.calls, [])

    def test_chat_text_only_path_unchanged(self):
        chunks = self._run([{'role': 'user', 'content': 'hello world'}])
        prefill = self.model.calls[0]
        self.assertIsNone(prefill['images'])
        self.assertIsNone(prefill['audios'])
        self.assertIsNone(prefill['mask'])
        self.assertTrue(all(chunk.content == '' or isinstance(chunk.content, str) for chunk in chunks))


class TestServiceMultimodal(unittest.TestCase):
    '''codon/utils/service.py：HTTP 侧的多模态请求'''

    def setUp(self):
        self.tokenizer = build_tokenizer()
        self.model = _FakeMultimodalLM(self.tokenizer.vocab_size)
        self.text_model = _TextOnlyLM(self.tokenizer.vocab_size)
        self.service = Service([
            ModelCard(model=self.model, tokenizer=self.tokenizer,
                      model_id='chord-test', owned='codon'),
            ModelCard(model=self.text_model, tokenizer=build_tokenizer(),
                      model_id='text-test', owned='codon'),
        ])

    def _request(self, model_id='chord-test', content=None, stream=False, max_tokens=3):
        request = ChatCompletionRequest(
            model=model_id,
            messages=[{'role': 'user', 'content': content}],
            stream=stream,
            max_tokens=max_tokens,
        )
        return asyncio.run(self.service.chat_completions(request))

    def test_non_streaming_multimodal_request(self):
        content = [
            {'type': 'text', 'text': 'hello'},
            {'type': 'image_url', 'image_url': {'url': png_data_url(32, 32)}},
            {'type': 'audio', 'audio': torch.randn(80, 64)},
        ]
        response = self._request(content=content)

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload['object'], 'chat.completion')
        self.assertIn(payload['choices'][0]['finish_reason'], ('stop', 'length'))
        self.assertIsInstance(payload['choices'][0]['message']['content'], str)

        prefill = self.model.calls[0]
        self.assertEqual(tuple(prefill['images'][0].shape), (3, 32, 32))
        self.assertEqual(tuple(prefill['audios'][0].shape), (80, 64))
        self.assertIsNotNone(prefill['mask'])

    def test_streaming_multimodal_request(self):
        content = [{'type': 'image', 'image': png_data_url(32, 32)}]
        response = self._request(content=content, stream=True)

        self.assertEqual(response.status_code, 200)

        async def collect():
            return [chunk async for chunk in response.body_iterator]

        events = asyncio.run(collect())
        body = ''.join(events)
        self.assertIn('data: [DONE]', body)
        self.assertEqual(tuple(self.model.calls[0]['images'][0].shape), (3, 32, 32))

    def test_unsupported_modality_returns_400(self):
        response = self._request(
            model_id='text-test',
            content=[{'type': 'image', 'image': png_data_url()}],
        )
        self.assertEqual(response.status_code, 400)
        payload = json.loads(response.body)
        self.assertEqual(payload['error']['code'], 'unsupported_modality')
        self.assertEqual(payload['error']['param'], 'content')
        self.assertEqual(self.text_model.calls, [])

    def test_invalid_media_returns_400(self):
        response = self._request(content=[
            {'type': 'image', 'image': 'data:image/png;base64,not-valid-base64!!'},
        ])
        self.assertEqual(response.status_code, 400)
        payload = json.loads(response.body)
        self.assertEqual(payload['error']['code'], 'invalid_content')

    def test_unknown_modality_type_video_returns_400(self):
        response = self._request(content=[
            {'type': 'video', 'video': 'data:video/mp4;base64,AAAA'},
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.body)['error']['code'], 'unsupported_modality')

    def test_text_request_still_works(self):
        response = self._request(content='hello world')
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.model.calls[0]['images'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
