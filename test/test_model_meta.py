import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import unittest

from codon.model.types.language import CausalLanguageModel, ModelMeta
from codon.motif.motif_a1 import MotifA1
from codon.motif.motif_a2 import MotifA2


class _PlainLM(CausalLanguageModel):
    def forward(self, input_ids, start_pos=0, past_key_values=None):
        raise NotImplementedError


class _ThinkingLM(_PlainLM):
    supports_thinking = True


class _FullLM(_ThinkingLM):
    supports_image = True
    supports_tool = True


class _MetaOverriddenLM(_PlainLM):
    meta = ModelMeta(supports_image=True, supports_thinking=True, supports_tool=True)


class _SpeechLM(_PlainLM):
    supports_audio = True
    audio_subtypes = '说话声'


class _MusicLM(_PlainLM):
    supports_audio = True
    audio_subtypes = ('music',)


class _AVLM(_SpeechLM):
    supports_image = True


class _NoAudioLM(_SpeechLM):
    supports_audio = False


class TestModelMeta(unittest.TestCase):
    def test_defaults_all_false(self):
        meta = ModelMeta()
        self.assertFalse(meta.supports_image)
        self.assertFalse(meta.supports_thinking)
        self.assertFalse(meta.supports_tool)
        self.assertFalse(meta.supports_audio)
        self.assertEqual(meta.audio_subtypes, ())
        self.assertEqual(meta.to_dict(), {
            'supports_image': False,
            'supports_thinking': False,
            'supports_tool': False,
            'supports_audio': False,
            'audio_subtypes': [],
        })

    def test_base_class_defaults(self):
        self.assertIsInstance(CausalLanguageModel.meta, ModelMeta)
        self.assertFalse(CausalLanguageModel.meta.supports_image)
        self.assertFalse(CausalLanguageModel.meta.supports_thinking)
        self.assertFalse(CausalLanguageModel.meta.supports_tool)
        self.assertFalse(CausalLanguageModel.meta.supports_audio)

    def test_subclass_inherits_defaults(self):
        self.assertEqual(_PlainLM.meta.to_dict(), ModelMeta().to_dict())

    def test_subclass_flag_merges_with_inherited(self):
        """子类只声明一个标记时，其余标记继承父类值。"""
        self.assertTrue(_ThinkingLM.meta.supports_thinking)
        self.assertFalse(_ThinkingLM.meta.supports_image)

        self.assertTrue(_FullLM.meta.supports_thinking)   # 继承 _ThinkingLM
        self.assertTrue(_FullLM.meta.supports_image)
        self.assertTrue(_FullLM.meta.supports_tool)

    def test_subclass_meta_does_not_mutate_parent(self):
        """子类的 meta 必须是独立实例，不能污染父类共享对象。"""
        self.assertIsNot(_ThinkingLM.meta, _PlainLM.meta)
        self.assertFalse(_PlainLM.meta.supports_thinking)
        self.assertIsNot(_FullLM.meta, _ThinkingLM.meta)

    def test_explicit_meta_override(self):
        self.assertEqual(_MetaOverriddenLM.meta.to_dict(), {
            'supports_image': True,
            'supports_thinking': True,
            'supports_tool': True,
            'supports_audio': False,
            'audio_subtypes': [],
        })

    def test_non_bool_flag_rejected(self):
        with self.assertRaises(TypeError):
            class _Bad(_PlainLM):
                supports_thinking = 'yes'

    def test_non_meta_object_rejected(self):
        with self.assertRaises(TypeError):
            class _Bad2(_PlainLM):
                meta = {'supports_thinking': True}

    def test_instance_helpers(self):
        class _Inst(_ThinkingLM):
            def __init__(self):
                super().__init__()

        inst = _Inst()
        self.assertTrue(inst.supports_thinking)
        self.assertFalse(inst.supports_image)
        self.assertTrue(inst.supports('thinking'))
        self.assertTrue(inst.supports('supports_thinking'))   # 兼容带前缀写法
        self.assertFalse(inst.supports('tool'))
        self.assertFalse(inst.supports('audio'))              # 未声明音频能力
        with self.assertRaises(ValueError):
            inst.supports('video')

    # ---- 音频能力 ----
    def test_audio_defaults_off(self):
        """未声明 supports_audio 时，音频能力与子类型都必须为空。"""
        self.assertFalse(ModelMeta().supports_audio)
        self.assertEqual(ModelMeta().audio_subtypes, ())
        self.assertFalse(_PlainLM.meta.supports_audio)
        self.assertEqual(_PlainLM.audio_subtypes, ())
        self.assertFalse(_PlainLM.meta.supports_audio_subtype('speech'))

    def test_audio_flag_without_subtype_is_general(self):
        """只声明 supports_audio = True 时按「通用」处理。"""
        meta = ModelMeta(supports_audio=True)
        self.assertTrue(meta.supports_audio)
        self.assertEqual(meta.audio_subtypes, ('general',))

    def test_audio_subtype_declaration(self):
        """子类可用单个字符串 / 集合声明子类型，实例上读到的是规范化结果。"""
        self.assertTrue(_SpeechLM.meta.supports_audio)
        self.assertEqual(_SpeechLM.meta.audio_subtypes, ('speech',))   # 中文别名 -> canonical
        self.assertEqual(_SpeechLM.audio_subtypes, ('speech',))
        self.assertEqual(_MusicLM.audio_subtypes, ('music',))

        class _MixedLM(_PlainLM):
            supports_audio = True
            audio_subtypes = ('音乐', 'speech', 'music')   # 去重 + 按 canonical 顺序

        self.assertEqual(_MixedLM.audio_subtypes, ('speech', 'music'))
        self.assertTrue(_MixedLM.meta.supports_audio_subtype('music'))
        self.assertTrue(_MixedLM.meta.supports_audio_subtype('音乐'))

    def test_audio_general_covers_other_subtypes(self):
        class _GeneralLM(_PlainLM):
            supports_audio = True
            audio_subtypes = ('speech', 'general')

        self.assertEqual(_GeneralLM.audio_subtypes, ('general',))
        self.assertTrue(_GeneralLM.meta.supports_audio_subtype('speech'))
        self.assertTrue(_GeneralLM.meta.supports_audio_subtype('music'))
        self.assertFalse(_MusicLM.meta.supports_audio_subtype('speech'))

    def test_audio_subtypes_cleared_when_audio_disabled(self):
        """supports_audio = False 时子类型被清空，且不污染父类。"""
        self.assertFalse(_NoAudioLM.meta.supports_audio)
        self.assertEqual(_NoAudioLM.audio_subtypes, ())
        self.assertTrue(_SpeechLM.meta.supports_audio)   # 父类不受影响
        self.assertEqual(_SpeechLM.audio_subtypes, ('speech',))

        self.assertEqual(ModelMeta(supports_audio=False, audio_subtypes='music').audio_subtypes, ())

    def test_audio_enabled_with_other_capabilities(self):
        """音频能力与既有能力标记互不干扰，可叠加声明。"""
        self.assertTrue(_AVLM.meta.supports_audio)
        self.assertEqual(_AVLM.audio_subtypes, ('speech',))
        self.assertTrue(_AVLM.meta.supports_image)

        inst = _AVLM()
        self.assertTrue(inst.supports('audio'))
        self.assertTrue(inst.supports('image'))
        self.assertTrue(inst.supports_audio)
        self.assertTrue(inst.supports_audio_subtype('说话声'))
        self.assertFalse(inst.supports_audio_subtype('music'))

    def test_audio_to_dict(self):
        self.assertEqual(_SpeechLM.meta.to_dict(), {
            'supports_image': False,
            'supports_thinking': False,
            'supports_tool': False,
            'supports_audio': True,
            'audio_subtypes': ['speech'],
        })

    def test_invalid_audio_subtype_rejected(self):
        with self.assertRaises(ValueError):
            ModelMeta(supports_audio=True, audio_subtypes='noise')

        with self.assertRaises(ValueError):
            class _BadAudio(_PlainLM):
                supports_audio = True
                audio_subtypes = ('speech', 'noise')

        with self.assertRaises(TypeError):
            ModelMeta(supports_audio=True, audio_subtypes=('speech', 1))

        with self.assertRaises(TypeError):
            ModelMeta(supports_audio=True, audio_subtypes=123)

    def test_motif_a1_supports_thinking(self):
        self.assertTrue(MotifA1.meta.supports_thinking)
        self.assertFalse(MotifA1.meta.supports_image)
        self.assertFalse(MotifA1.meta.supports_tool)
        self.assertFalse(MotifA1.meta.supports_audio)
        self.assertEqual(MotifA1.audio_subtypes, ())
        self.assertIsInstance(MotifA1.meta, ModelMeta)

    def test_motif_a2_defaults_false(self):
        """未声明能力的模型（MotifA2）必须保持全 False。"""
        self.assertEqual(MotifA2.meta.to_dict(), ModelMeta().to_dict())


if __name__ == '__main__':
    unittest.main(verbosity=2)
