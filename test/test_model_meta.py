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


class TestModelMeta(unittest.TestCase):
    def test_defaults_all_false(self):
        meta = ModelMeta()
        self.assertFalse(meta.supports_image)
        self.assertFalse(meta.supports_thinking)
        self.assertFalse(meta.supports_tool)
        self.assertEqual(meta.to_dict(), {
            'supports_image': False,
            'supports_thinking': False,
            'supports_tool': False,
        })

    def test_base_class_defaults(self):
        self.assertIsInstance(CausalLanguageModel.meta, ModelMeta)
        self.assertFalse(CausalLanguageModel.meta.supports_image)
        self.assertFalse(CausalLanguageModel.meta.supports_thinking)
        self.assertFalse(CausalLanguageModel.meta.supports_tool)

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
        with self.assertRaises(ValueError):
            inst.supports('audio')

    def test_motif_a1_supports_thinking(self):
        self.assertTrue(MotifA1.meta.supports_thinking)
        self.assertFalse(MotifA1.meta.supports_image)
        self.assertFalse(MotifA1.meta.supports_tool)
        self.assertIsInstance(MotifA1.meta, ModelMeta)

    def test_motif_a2_defaults_false(self):
        """未声明能力的模型（MotifA2）必须保持全 False。"""
        self.assertEqual(MotifA2.meta.to_dict(), ModelMeta().to_dict())


if __name__ == '__main__':
    unittest.main(verbosity=2)
