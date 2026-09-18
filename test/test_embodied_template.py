'''
具身模板链路测试：codon.j2 新增结构 token -> Session 展开 -> MotifChord 就地注入。

覆盖：
    codon.j2    grounding / action / proprio / tactile / force / camera / trajectory /
                waypoint / video 帧 / <|environment|> role
    Session     <|action_start|><|action_pad|><|action_end|> 三连按张量行数展开，
                to_tensors() 给出 *_patch_indices 与 actions/proprios/tactiles
    MotifChord  x[b, action_patch_indices[b]] = 动作特征（序列长度不变、位置不移位）
'''

import os
import sys
import unittest

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from unittest import mock

from codon.res import LM
from codon.utils.media import UnsupportedModalityError, has_video_frames, normalize_messages
from codon.utils.session import Session
from codon.utils.tokens import PackedTokenizer

import codon.motif.chord.model as chord_model
from codon.motif.chord.config import MotifChordConfig
from codon.motif.chord.model import MotifChord


# ---------------------------------------------------------------- 词表

SPECIALS = [
    '<|unk|>', '<|pad|>', '<|im_start|>', '<|im_end|>', '<|system|>', '<|user|>',
    '<|model|>', '<|environment|>', '<|tool_response|>', '<|tool_name_divider|>',
    '<|thought_start|>', '<|thought_end|>',
    '<|effort_low|>', '<|effort_high|>', '<|effort_max|>',
    '<|modality_image_start|>', '<|modality_image_pad|>', '<|modality_image_end|>',
    '<|modality_audio_start|>', '<|modality_audio_pad|>', '<|modality_audio_end|>',
    '<|modality_video_start|>', '<|modality_video_pad|>', '<|modality_video_end|>',
    '<|frame_start|>', '<|frame_end|>', '<|frame_sep|>',
    '<|timestamp_start|>', '<|timestamp_end|>',
    '<|action_start|>', '<|action_pad|>', '<|action_end|>',
    '<|action_chunk_start|>', '<|action_sep|>', '<|action_chunk_end|>',
    '<|action_terminate|>', '<|action_continue|>',
    '<|proprio_start|>', '<|proprio_pad|>', '<|proprio_end|>',
    '<|tactile_start|>', '<|tactile_pad|>', '<|tactile_end|>',
    '<|force_start|>', '<|force_end|>',
    '<|camera_start|>', '<|camera_end|>', '<|camera_id|>',
    '<|trajectory_start|>', '<|trajectory_end|>',
    '<|waypoint_start|>', '<|waypoint_end|>',
    '<|bbox_start|>', '<|bbox_end|>', '<|point_start|>', '<|point_end|>',
    '<|ref_start|>', '<|ref_end|>', '<|grasp_start|>', '<|grasp_end|>',
    '[unused_42]',
]
WORDS = ['hello', 'world']


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


def build_session(**kwargs) -> Session:
    return Session(build_tokenizer(), patch_size=4, audio_pool_stride=2, **kwargs)


def render(tokenizer, messages, **kwargs) -> str:
    kwargs.setdefault('tokenize', False)
    return tokenizer.fast_tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, **kwargs
    )


# ---------------------------------------------------------------- 模板渲染

class TestTemplateBlocks(unittest.TestCase):

    def setUp(self):
        self.tok = build_tokenizer()

    def test_grounding_blocks(self):
        text = render(self.tok, [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'grab '},
            {'type': 'ref', 'ref': 'the red cup'},
            {'type': 'bbox', 'bbox': '[0.123,0.456,0.789,0.900]'},
            {'type': 'point', 'point': '[0.512,0.334]'},
            {'type': 'grasp', 'grasp': '[0.51,0.33]'},
        ]}])
        self.assertEqual(
            text,
            '<|im_start|><|user|>grab <|ref_start|>the red cup<|ref_end|>'
            '<|bbox_start|>[0.123,0.456,0.789,0.900]<|bbox_end|>'
            '<|point_start|>[0.512,0.334]<|point_end|>'
            '<|grasp_start|>[0.51,0.33]<|grasp_end|><|im_end|>'
        )

    def test_ref_item_with_text_key_is_wrapped(self):
        '''{'type': 'ref', 'text': ...} 必须被 ref 包住，而不是退化成裸文本。'''
        text = render(self.tok, [{'role': 'user', 'content': [
            {'type': 'ref', 'text': 'the blue box'}]}])
        self.assertIn('<|ref_start|>the blue box<|ref_end|>', text)

    def test_action_triples(self):
        text = render(self.tok, [{'role': 'model', 'content': [
            {'type': 'action'},
            {'type': 'action_chunk', 'actions': [1, 2], 'flag': 'terminate'},
        ]}])
        self.assertIn(
            '<|action_start|><|action_pad|><|action_end|>'
            '<|action_chunk_start|>'
            '<|action_start|><|action_pad|><|action_end|>'
            '<|action_sep|>'
            '<|action_start|><|action_pad|><|action_end|>'
            '<|action_terminate|><|action_chunk_end|>',
            text,
        )

    def test_action_chunk_tensor_is_single_step(self):
        '''张量是「一个 block」，不是多个 step：一次三连。'''
        text = render(self.tok, [{'role': 'model', 'content': [
            {'type': 'action_chunk', 'actions': torch.zeros(4, 3), 'flag': 'continue'}]}])
        self.assertEqual(text.count('<|action_pad|>'), 1)
        self.assertIn('<|action_continue|><|action_chunk_end|>', text)

    def test_observation_blocks(self):
        text = render(self.tok, [{'role': 'model', 'content': [
            {'type': 'proprio'},
            {'type': 'tactile'},
            {'type': 'force', 'force': '12.30'},
            {'type': 'camera', 'camera': 'front', 'id': 0},
            {'type': 'trajectory', 'trajectory': '[0,1],[2,3]'},
            {'type': 'waypoint', 'waypoint': '[1.0,2.0]'},
        ]}])
        self.assertIn('<|proprio_start|><|proprio_pad|><|proprio_end|>', text)
        self.assertIn('<|tactile_start|><|tactile_pad|><|tactile_end|>', text)
        self.assertIn('<|force_start|>12.30<|force_end|>', text)
        # camera_id 的 0 不能被 default(..., true) 吞掉
        self.assertIn('<|camera_start|>front<|camera_id|>0<|camera_end|>', text)
        self.assertIn('<|trajectory_start|>[0,1],[2,3]<|trajectory_end|>', text)
        self.assertIn('<|waypoint_start|>[1.0,2.0]<|waypoint_end|>', text)

    def test_environment_role_is_its_own_turn(self):
        text = render(self.tok, [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'go'},
            {'role': 'environment', 'content': 'collision detected'},
        ], role_strict=True)
        self.assertEqual(
            text,
            '<|im_start|><|system|>sys<|im_end|>'
            '<|im_start|><|user|>go<|im_end|>'
            '<|im_start|><|environment|>collision detected<|im_end|>'
        )

    def test_environment_first_message_is_not_folded_into_header(self):
        text = render(self.tok, [
            {'role': 'environment', 'content': 'obs'},
            {'role': 'user', 'content': 'go'},
        ])
        self.assertEqual(
            text,
            '<|im_start|><|environment|>obs<|im_end|>'
            '<|im_start|><|user|>go<|im_end|>'
        )

    def test_video_frames_render(self):
        '''逐帧时间戳：帧自带 timestamp 优先，否则取 timestamps[帧下标]。'''
        text = render(self.tok, [{'role': 'user', 'content': [
            {'type': 'video', 'video': 'x.mp4',
             'frames': [{'image': 'a', 'timestamp': '0.00'}, {'image': 'b'}],
             'timestamps': ['0.00', '9.99']},
        ]}], image_capab=True, video_capab=True)
        self.assertEqual(
            text,
            '<|im_start|><|user|><|modality_video_start|>'
            '<|frame_start|><|timestamp_start|>0.00<|timestamp_end|>'
            '<|modality_image_start|><|modality_image_pad|><|modality_image_end|>'
            '<|frame_end|><|frame_sep|>'
            '<|frame_start|><|timestamp_start|>9.99<|timestamp_end|>'
            '<|modality_image_start|><|modality_image_pad|><|modality_image_end|>'
            '<|frame_end|>'
            '<|modality_video_end|><|im_end|>'
        )

    def test_video_without_frames_keeps_single_placeholder(self):
        text = render(self.tok, [{'role': 'user', 'content': [
            {'type': 'video', 'video': 'x.mp4'}]}], video_capab=True)
        self.assertIn(
            '<|modality_video_start|><|modality_video_pad|><|modality_video_end|>', text)

    def test_structured_payload_is_escaped_for_text_only(self):
        '''safe_rules 只作用于 text item：结构化字段里的方括号必须原样保留。'''
        tok = build_tokenizer()
        ids = tok.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'bbox', 'bbox': '[0.1,0.2]'}]}],
            add_generation_prompt=False,
        )
        self.assertIn('<|bbox_start|>', tok.decode(ids))

    def test_literal_tokens_in_text_are_neutralized(self):
        '''调用方不能靠 text 注入结构 token（会被 safe_rules 打散）。'''
        tok = build_tokenizer()
        ids = tok.apply_chat_template(
            [{'role': 'user', 'content': 'evil <|bbox_start|>'}],
            add_generation_prompt=False,
        )
        self.assertNotIn(tok.token_to_id('<|bbox_start|>'), ids)

    def test_system_message_blocks_respect_strict(self):
        with self.assertRaises(Exception):
            render(self.tok, [{'role': 'system', 'content': [
                {'type': 'bbox', 'bbox': '[0.1]'}]}], attachment_strict=True)


# ---------------------------------------------------------------- Session 展开

class TestSessionExpansion(unittest.TestCase):

    def test_action_triple_expands_to_tensor_rows(self):
        session = build_session()
        action = torch.randn(5, 7)
        session.add_message({'role': 'model', 'content': [
            {'type': 'text', 'text': 'ok'},
            {'type': 'action', 'action': action},
        ]}, mask='all')
        ids = session.input_ids
        pad_id = session.token_id('action_pad')
        self.assertEqual(ids.count(pad_id), 5)

        tensors = session.to_tensors()
        self.assertEqual(tensors['action_patch_indices'].tolist(),
                         list(range(ids.index(pad_id), ids.index(pad_id) + 5)))
        self.assertEqual(len(tensors['actions']), 1)
        self.assertTrue(torch.equal(tensors['actions'][0], action))

    def test_action_chunk_list_gives_one_triple_per_step(self):
        session = build_session()
        steps = [torch.randn(7) for _ in range(3)]
        session.add_message({'role': 'model', 'content': [
            {'type': 'action_chunk', 'actions': steps},
        ]}, mask='all')
        ids = session.input_ids
        pad_id = session.token_id('action_pad')
        self.assertEqual(ids.count(pad_id), 3)

        tensors = session.to_tensors()
        self.assertEqual(tensors['action_patch_indices'].numel(), 3)
        self.assertEqual(len(tensors['actions']), 3)

    def test_proprio_and_tactile_indices(self):
        session = build_session()
        session.add_message({'role': 'environment', 'content': [
            {'type': 'proprio', 'proprio': torch.randn(2, 7)},
            {'type': 'tactile', 'tactile': torch.randn(4, 32)},
        ]})
        tensors = session.to_tensors()
        self.assertEqual(tensors['proprio_patch_indices'].numel(), 2)
        self.assertEqual(tensors['tactile_patch_indices'].numel(), 4)
        self.assertEqual(len(tensors['proprios']), 1)
        self.assertEqual(len(tensors['tactiles']), 1)
        # 环境反馈默认不进 loss
        self.assertTrue(all(session.ignore_mask))

    def test_action_pads_are_masked_out_of_loss(self):
        session = build_session()
        session.add_message({'role': 'model', 'content': [
            {'type': 'text', 'text': 'ok'},
            {'type': 'action', 'action': torch.randn(3, 7)},
        ]})
        pad_id = session.token_id('action_pad')
        labels = session.labels
        for idx, tid in enumerate(session.input_ids):
            if tid == pad_id:
                self.assertEqual(labels[idx], -100)

    def test_block_count_mismatch_raises(self):
        session = build_session()
        with self.assertRaises(ValueError):
            session.add_message({'role': 'model', 'content': [
                {'type': 'action', 'action': torch.randn(7)},
                {'type': 'action'},
            ]})

    def test_video_frames_collected_as_images(self):
        session = build_session(image_capab=True, video_capab=True)
        frames = [torch.randn(3, 8, 8), torch.randn(3, 8, 8)]
        session.add_message({'role': 'user', 'content': [
            {'type': 'video', 'video': 'x.mp4',
             'frames': [{'image': frames[0], 'timestamp': '0.00'}, {'image': frames[1]}]},
        ]})
        tensors = session.to_tensors()
        self.assertEqual(len(tensors['images']), 2)
        # patch_size=4、8x8 帧 -> 每帧 4 个占位符
        self.assertEqual(tensors['image_patch_indices'].numel(), 8)

    def test_video_frames_auto_enable_image_capability(self):
        '''video + frames 复用图像塔，Session 必须自己把 image_capab 打开。'''
        session = build_session(video_capab=True)      # image_capab 留 None（自动）
        session.add_message({'role': 'user', 'content': [
            {'type': 'video', 'video': 'x.mp4',
             'frames': [{'image': torch.randn(3, 8, 8)}]},
        ]})
        self.assertIn('<|frame_start|>', session.decode())


# ---------------------------------------------------------------- media 规整

class TestMediaNormalization(unittest.TestCase):

    def test_frames_are_decoded_to_tensors(self):
        messages = normalize_messages(
            [{'role': 'user', 'content': [
                {'type': 'video', 'video': 'x.mp4',
                 'frames': [{'image': torch.zeros(3, 4, 4), 'timestamp': '0.00'},
                            {'image': torch.ones(3, 4, 4)}]},
            ]}],
            image_capab=True,
        )
        frames = messages[0]['content'][0]['frames']
        self.assertEqual(len(frames), 2)
        self.assertIsInstance(frames[0]['image'], torch.Tensor)
        self.assertEqual(frames[0]['timestamp'], '0.00')

    def test_frames_without_image_capability_raise(self):
        with self.assertRaises(UnsupportedModalityError):
            normalize_messages(
                [{'role': 'user', 'content': [
                    {'type': 'video', 'video': 'x.mp4',
                     'frames': [{'image': torch.zeros(3, 4, 4)}]},
                ]}],
                image_capab=False,
            )

    def test_raw_video_still_needs_video_tower(self):
        with self.assertRaises(UnsupportedModalityError):
            normalize_messages(
                [{'role': 'user', 'content': [{'type': 'video', 'video': 'x.mp4'}]}],
                image_capab=True, video_capab=False,
            )

    def test_has_video_frames(self):
        self.assertTrue(has_video_frames([{'content': [
            {'type': 'video', 'video': 'x', 'frames': [{'image': 1}]}]}]))
        self.assertFalse(has_video_frames([{'content': [
            {'type': 'video', 'video': 'x'}]}]))

    def test_action_items_pass_through_untouched(self):
        action = torch.randn(3, 7)
        messages = normalize_messages(
            [{'role': 'model', 'content': [{'type': 'action', 'action': action}]}],
        )
        self.assertIs(messages[0]['content'][0]['action'], action)


if __name__ == '__main__':
    unittest.main()


# ---------------------------------------------------------------- 模型注入

ACTION_ROWS = 3
ACTION_DIM = 4
ACTION_TARGET = 11
PROPRIO_DIM = 5
PROPRIO_TARGET = 12
TACTILE_DIM = 6
TACTILE_TARGET = 13

_FAKE_WEIGHTS = {}


def _make_fake_fixed_projector(key: str, num_tokens: int, target_id: int):
    '''假冻结塔（视觉 / 音频）：无论输入是什么都返回 num_tokens 行 token_emb[target_id]。'''

    class _FakeProjector(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.num_tokens = num_tokens
            self.target_id = target_id

        def forward(self, x):
            weight = _FAKE_WEIGHTS[key]
            emb = weight[self.target_id]
            return emb.view(1, 1, -1).expand(x.shape[0], self.num_tokens, -1)

    return _FakeProjector


def _make_fake_vector_projector(key: str, targets_by_dim: dict):
    '''假向量塔：行数由输入行数决定，目标 token 由 in_features（模态维度）决定。'''

    class _FakeVectorProjector(torch.nn.Module):
        def __init__(self, config, in_features):
            super().__init__()
            self.config = config
            self.in_features = int(in_features)
            self.target_id = targets_by_dim[self.in_features]

        def forward(self, x):
            weight = _FAKE_WEIGHTS[key]
            rows = x.shape[1] if x.dim() >= 3 else x.shape[0]
            return weight[self.target_id].view(1, 1, -1).expand(x.shape[0], rows, -1)

    return _FakeVectorProjector


def build_model() -> MotifChord:
    config = MotifChordConfig(
        vocab_size=64,
        model_dim=32,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.0,
        gdn_head_k_dim=16,
        vision_dim=8,
        audio_dim=8,
        audio_pool_stride=2,
        action_dim=ACTION_DIM,
        proprio_dim=PROPRIO_DIM,
        tactile_dim=TACTILE_DIM,
    )
    vision_cls = _make_fake_fixed_projector('vision', 6, 5)
    audio_cls = _make_fake_fixed_projector('audio', 4, 7)
    vector_cls = _make_fake_vector_projector('vector', {
        ACTION_DIM: ACTION_TARGET,
        PROPRIO_DIM: PROPRIO_TARGET,
        TACTILE_DIM: TACTILE_TARGET,
    })

    with mock.patch.object(chord_model, 'MotifChordVisionProjector', vision_cls), \
         mock.patch.object(chord_model, 'MotifChordAudioProjector', audio_cls), \
         mock.patch.object(chord_model, 'MotifChordVectorProjector', vector_cls):
        model = MotifChord(config)

    _FAKE_WEIGHTS['vision'] = model.token_emb.weight
    _FAKE_WEIGHTS['audio'] = model.token_emb.weight
    _FAKE_WEIGHTS['vector'] = model.token_emb.weight
    model.eval()
    return model


class TestVectorInjection(unittest.TestCase):

    def setUp(self):
        self.model = build_model()

    def _tensors_for(self, content, **kwargs):
        session = build_session(**kwargs)
        session.add_message({'role': 'model', 'content': content}, mask='all')
        return session, session.to_tensors()

    def test_action_injection_equals_token_replacement(self):
        session, tensors = self._tensors_for([
            {'type': 'text', 'text': 'ok'},
            {'type': 'action', 'action': torch.randn(ACTION_ROWS, ACTION_DIM)},
        ])
        ids = tensors['input_ids'].unsqueeze(0)
        indices = tensors['action_patch_indices'].unsqueeze(0)

        with torch.no_grad():
            injected = self.model(
                input_ids=ids, actions=tensors['actions'],
                action_patch_indices=indices,
            ).logits
            replaced_ids = ids.clone()
            replaced_ids[0, indices[0]] = ACTION_TARGET
            replaced = self.model(input_ids=replaced_ids).logits
            without = self.model(input_ids=ids).logits

        self.assertEqual(injected.shape, (1, ids.shape[1], self.model.config.vocab_size))
        self.assertTrue(torch.allclose(injected, replaced, atol=1e-6))
        self.assertFalse(torch.allclose(injected, without))

    def test_action_patch_id_scan_matches_explicit_indices(self):
        session, tensors = self._tensors_for([
            {'type': 'action', 'action': torch.randn(ACTION_ROWS, ACTION_DIM)},
        ])
        ids = tensors['input_ids'].unsqueeze(0)
        pad_id = session.token_id('action_pad')

        with torch.no_grad():
            explicit = self.model(
                input_ids=ids, actions=tensors['actions'],
                action_patch_indices=tensors['action_patch_indices'].unsqueeze(0),
            ).logits
            scanned = self.model(
                input_ids=ids, actions=tensors['actions'], action_patch_id=pad_id,
            ).logits

        self.assertTrue(torch.allclose(explicit, scanned))

    def test_proprio_and_tactile_injection(self):
        session, tensors = self._tensors_for([
            {'type': 'proprio', 'proprio': torch.randn(2, PROPRIO_DIM)},
            {'type': 'tactile', 'tactile': torch.randn(TACTILE_DIM)},
        ])
        ids = tensors['input_ids'].unsqueeze(0)

        with torch.no_grad():
            out = self.model(
                input_ids=ids,
                proprios=tensors['proprios'],
                proprio_patch_indices=tensors['proprio_patch_indices'].unsqueeze(0),
                tactiles=tensors['tactiles'],
                tactile_patch_indices=tensors['tactile_patch_indices'].unsqueeze(0),
            ).logits
            replaced_ids = ids.clone()
            replaced_ids[0, tensors['proprio_patch_indices']] = PROPRIO_TARGET
            replaced_ids[0, tensors['tactile_patch_indices']] = TACTILE_TARGET
            replaced = self.model(input_ids=replaced_ids).logits

        self.assertTrue(torch.allclose(out, replaced, atol=1e-6))

    def test_missing_actions_for_placeholders_raises(self):
        session, tensors = self._tensors_for([
            {'type': 'action', 'action': torch.randn(2, ACTION_DIM)},
        ])
        ids = tensors['input_ids'].unsqueeze(0)
        with self.assertRaises(ValueError):
            self.model(input_ids=ids,
                       action_patch_indices=tensors['action_patch_indices'].unsqueeze(0))

    def test_row_count_mismatch_raises(self):
        session, tensors = self._tensors_for([
            {'type': 'action', 'action': torch.randn(2, ACTION_DIM)},
        ])
        ids = tensors['input_ids'].unsqueeze(0)
        with self.assertRaises(ValueError):
            self.model(
                input_ids=ids,
                actions=[torch.randn(5, ACTION_DIM)],
                action_patch_indices=tensors['action_patch_indices'].unsqueeze(0),
            )

    def test_vector_projector_gets_config_dims(self):
        '''三个向量塔的输入宽度必须来自 config 的对应维度。'''
        config = self.model.config
        self.assertEqual(config.action_dim, ACTION_DIM)
        self.assertEqual(config.proprio_dim, PROPRIO_DIM)
        self.assertEqual(config.tactile_dim, TACTILE_DIM)
        self.assertEqual(self.model.action_proj.in_features, ACTION_DIM)
        self.assertEqual(self.model.proprio_proj.in_features, PROPRIO_DIM)
        self.assertEqual(self.model.tactile_proj.in_features, TACTILE_DIM)


if __name__ == '__main__':
    unittest.main()
