import asyncio
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import unittest

from codon.utils.service import ChatCompletionRequest, ModelCard, Service


class _StubModel:
    '''只需要能放进 ModelCard：response_format 校验发生在真正生成之前。'''

    def eval(self):
        return self


class _StubTokenizer:
    vocab_size = 0


class TestServiceResponseFormat(unittest.TestCase):
    def setUp(self):
        self.service = Service([
            ModelCard(
                model=_StubModel(),
                tokenizer=_StubTokenizer(),
                model_id='motif-test',
                owned='codon',
            )
        ])

    def request(self, response_format):
        request = ChatCompletionRequest(
            model='motif-test',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=response_format,
        )
        return asyncio.run(self.service.chat_completions(request))

    def test_invalid_response_format_returns_400(self):
        cases = [
            {'type': 'yaml'},
            {'type': 'json_schema'},
            {'type': 'json_schema', 'json_schema': {'schema': {'anyOf': [{'type': 'string'}]}}},
            {'type': 'json_object', 'schema': {'type': 'string', 'pattern': '^a'}},
        ]
        for response_format in cases:
            response = self.request(response_format)
            self.assertEqual(response.status_code, 400, response_format)
            payload = json.loads(response.body)
            self.assertEqual(payload['error']['code'], 'invalid_response_format')
            self.assertEqual(payload['error']['param'], 'response_format')

    def test_unknown_model_returns_404(self):
        request = ChatCompletionRequest(
            model='nope',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format={'type': 'json_object'},
        )
        response = asyncio.run(self.service.chat_completions(request))
        self.assertEqual(response.status_code, 404)

    def test_response_format_is_optional(self):
        request = ChatCompletionRequest(
            model='motif-test',
            messages=[{'role': 'user', 'content': 'hi'}],
        )
        self.assertIsNone(request.response_format)


if __name__ == '__main__':
    unittest.main(verbosity=2)
