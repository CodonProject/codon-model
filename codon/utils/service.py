import asyncio
import time
import uuid
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple, Union

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from codon.model.types.language import CausalLanguageModel
from codon.model.grammar import parse_response_format
from codon.utils.tokens import PackedTokenizer
from codon.utils.media import (
    UnsupportedModalityError,
    modal_config,
    model_modalities,
    normalize_messages,
)


@dataclass
class ModelCard:
    '''
    Model information card for registering models to the service.

    Attributes:
        model (CausalLanguageModel): The causal language model instance.
        tokenizer (PackedTokenizer): The tokenizer associated with the model.
        model_id (str): The unique identifier for the model.
        owned (str): Owner identifier of the model.
    '''
    model: CausalLanguageModel
    tokenizer: PackedTokenizer
    model_id: str
    owned: str


class ChatMessage(BaseModel):
    '''
    A class representing a single message in a chat history.

    Attributes:
        role (str): Role of the sender ('system', 'user', 'assistant').
        content (Union[str, List[Dict[str, Any]]]): Message content。字符串是纯文本；列表则是
            codon.j2 风格的多模态内容，媒体项在服务端解码成张量：

                [{'type': 'text', 'text': '...'},
                 {'type': 'image', 'image': '<data URL / base64 / 路径 / http URL>'},
                 {'type': 'audio', 'audio': '<mel 张量载荷 / .npy / .pt / wav>'}]

            也接受 OpenAI 风格写法：`{'type': 'image_url', 'image_url': {'url': ...}}` 与
            `{'type': 'input_audio', 'input_audio': {'data': '<base64>', 'format': 'wav'}}`。
    '''
    role: str
    content: Union[str, List[Dict[str, Any]]]

    model_config = {
        'extra': 'allow'
    }


class ChatCompletionRequest(BaseModel):
    '''
    A class representing a request to the chat completions endpoint.

    Attributes:
        model (str): The ID of the model to use.
        messages (List[ChatMessage]): The chat history messages.
        temperature (float): The sampling temperature. Defaults to 0.7.
        top_p (Optional[float]): The nucleus sampling probability.
        top_k (Optional[int]): The top-k sampling threshold.
        repetition_penalty (float): Repetition penalty parameter. Defaults to 1.15.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 100.
        stream (bool): Whether to stream back partial progress. Defaults to False.
        response_format (Optional[Dict[str, Any]]): 强制 JSON 输出（OpenAI 兼容）：
            {'type': 'json_object'} / {'type': 'json'} / {'type': 'json_array'} /
            {'type': 'json_schema', 'json_schema': {'name': str, 'schema': {...}, 'strict': bool}} /
            {'type': 'text'}。默认 None，即自由文本。
    '''
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: float = 1.15
    max_tokens: int = 1024
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = None

    model_config = {
        'extra': 'allow'
    }


class Service:
    '''
    OpenAI-compatible FastAPI service wrapper for CausalLanguageModel.

    Attributes:
        models (Dict[str, ModelCard]): Registered model cards indexed by model_id.
        locks (Dict[str, asyncio.Lock]): Mutex locks for each model to ensure concurrency safety.
        app (FastAPI): The underlying FastAPI application instance.
    '''

    def __init__(self, models: List[ModelCard]) -> None:
        '''
        Initializes the Service with a list of model cards.

        Args:
            models (List[ModelCard]): A list of ModelCard instances to host.
        '''
        self.models = {card.model_id: card for card in models}
        self.locks = {card.model_id: asyncio.Lock() for card in models}
        self.app = FastAPI(title='Codon Model Service')

        # Add CORS middleware to support standard client connections
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )

        self._register_routes()

    def _register_routes(self) -> None:
        '''Registers FastAPI routes for OpenAI compatibility.'''
        self.app.get('/v1/models')(self.list_models)
        self.app.get('/models')(self.list_models)
        self.app.post('/v1/chat/completions')(self.chat_completions)
        self.app.post('/chat/completions')(self.chat_completions)
        self.app.post('/v1/responses')(self.chat_completions)

    @staticmethod
    def _safe_next(iterator: Any) -> Optional[Any]:
        '''
        Safely retrieves the next item from an iterator, catching StopIteration.

        Args:
            iterator (Any): The iterator to get the next item from.

        Returns:
            Optional[Any]: The next item, or None if StopIteration is raised.
        '''
        try:
            return next(iterator)
        except StopIteration:
            return None

    @staticmethod
    def _error(status_code: int, message: str, code: str, param: str) -> JSONResponse:
        '''OpenAI 风格的错误响应。'''
        return JSONResponse(
            status_code=status_code,
            content={
                'error': {
                    'message': message,
                    'type': 'invalid_request_error',
                    'param': param,
                    'code': code,
                }
            }
        )

    @staticmethod
    def _format_messages(model: CausalLanguageModel, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        '''
        把请求消息规整成 `chat()` 能吃的格式：文本原样保留，媒体载荷解码成张量。

        Raises:
            UnsupportedModalityError: 消息带了模型没有声明的模态（图片 / 音频 / 视频）。
            ValueError / TypeError / OSError: 媒体载荷无法解码。
        '''
        raw = [{'role': msg.role, 'content': msg.content} for msg in messages]
        capabilities = model_modalities(model)
        _, _, num_mel_bins = modal_config(model)
        return normalize_messages(
            raw,
            image_capab=capabilities['image'],
            audio_capab=capabilities['audio'],
            video_capab=capabilities['video'],
            num_mel_bins=num_mel_bins,
        )

    async def list_models(self) -> JSONResponse:
        '''
        Lists the available models registered in this service.

        Returns:
            JSONResponse: Standard list of model information.
        '''
        data = []
        for model_id, card in self.models.items():
            meta = getattr(card.model, 'meta', None)
            data.append({
                'id': model_id,
                'object': 'model',
                'created': int(time.time()),
                'owned_by': card.owned,
                'capabilities': meta.to_dict() if meta is not None else None,
            })
        return JSONResponse(content={'object': 'list', 'data': data})

    def _make_chunk(
        self,
        request_id: str,
        model_id: str,
        content: str,
        reasoning_content: str,
        created_time: int,
        finish_reason: Optional[str]
    ) -> Dict[str, Any]:
        '''
        Creates an OpenAI-compatible chat completion chunk dictionary supporting reasoning_content.

        Args:
            request_id (str): Unique request identifier.
            model_id (str): Model identifier.
            content (str): The token text delta.
            reasoning_content (str): The reasoning token text delta.
            created_time (int): Generation timestamp.
            finish_reason (Optional[str]): Why the generation finished, if complete.

        Returns:
            Dict[str, Any]: A structured chunk response.
        '''
        delta = {}
        if content:
            delta['content'] = content
        if reasoning_content:
            delta['reasoning_content'] = reasoning_content

        return {
            'id': request_id,
            'object': 'chat.completion.chunk',
            'created': created_time,
            'model': model_id,
            'choices': [
                {
                    'index': 0,
                    'delta': delta,
                    'logprobs': None,
                    'finish_reason': finish_reason
                }
            ]
        }

    async def _stream_generator(
        self,
        request_id: str,
        model_id: str,
        model: CausalLanguageModel,
        tokenizer: PackedTokenizer,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        created_time: int,
        response_format: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        '''
        Asynchronous generator that yields OpenAI-compatible SSE events for chat completion streaming.
        Uses the official 'chat' function from 'codon.utils.generate' for maximum accuracy and speed.
        '''
        async with self.locks[model_id]:
            # Run generator loop inside an executor thread to keep FastAPI responsive
            def _blocking_generator():
                from codon.utils.generate import chat
                for chunk in chat(
                    model=model,
                    tokenizer=tokenizer,
                    device=next(model.parameters()).device,
                    messages=messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    response_format=response_format
                ):
                    yield chunk

            loop = asyncio.get_running_loop()
            iterator = _blocking_generator()

            finish_reason = 'stop'
            while True:
                chunk = await loop.run_in_executor(None, self._safe_next, iterator)
                if chunk is None:
                    break

                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason   # 最后一块只报结束原因
                    continue

                content = '' if chunk.is_cot else chunk.content
                reasoning_content = chunk.content if chunk.is_cot else ''

                if content or reasoning_content:
                    yield f'data: {json.dumps(self._make_chunk(request_id, model_id, content, reasoning_content, created_time, None))}\n\n'

            yield f"data: {json.dumps(self._make_chunk(request_id, model_id, '', '', created_time, finish_reason))}\n\n"
            yield 'data: [DONE]\n\n'

    async def chat_completions(self, request: ChatCompletionRequest) -> Any:
        '''
        Handles standard and streaming OpenAI chat completion requests.

        Args:
            request (ChatCompletionRequest): The formatted incoming request parameters.

        Returns:
            Union[JSONResponse, StreamingResponse]: Streaming SSE data or fully aggregated JSON.
        '''
        model_id = request.model
        if model_id not in self.models:
            return self._error(
                404,
                f"Model '{model_id}' not found.",
                'model_not_found',
                'model',
            )

        card = self.models[model_id]
        model = card.model
        tokenizer = card.tokenizer

        # Validate response_format early so bad schemas come back as 400 instead of 500
        try:
            parse_response_format(request.response_format)
        except (TypeError, ValueError) as exc:
            return self._error(
                400,
                f'Invalid response_format: {exc}',
                'invalid_response_format',
                'response_format',
            )

        # 多模态内容：媒体载荷就地解码成张量，并校验模型能力（不支持就 400，而不是喂 unk token）
        try:
            formatted_messages = self._format_messages(model, request.messages)
        except UnsupportedModalityError as exc:
            return self._error(400, str(exc), 'unsupported_modality', 'content')
        except (TypeError, ValueError, OSError, RuntimeError) as exc:
            return self._error(400, f'Invalid message content: {exc}', 'invalid_content', 'content')

        request_id = f'chatcmpl-{uuid.uuid4()}'
        created_time = int(time.time())

        if request.stream:
            return StreamingResponse(
                self._stream_generator(
                    request_id=request_id,
                    model_id=model_id,
                    model=model,
                    tokenizer=tokenizer,
                    messages=formatted_messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    created_time=created_time,
                    response_format=request.response_format
                ),
                media_type='text/event-stream'
            )

        # Non-streaming implementation optimized for singlelock using chat helper
        async with self.locks[model_id]:
            def _blocking_generate() -> Tuple[str, str, int, str]:
                from codon.utils.generate import chat
                content_accum = []
                reasoning_accum = []
                total_tokens = 0
                finish_reason = 'stop'
                for chunk in chat(
                    model=model,
                    tokenizer=tokenizer,
                    device=next(model.parameters()).device,
                    messages=formatted_messages,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    response_format=request.response_format
                ):
                    if chunk.finish_reason is not None:
                        finish_reason = chunk.finish_reason
                        continue
                    if chunk.content:
                        if chunk.is_cot:
                            reasoning_accum.append(chunk.content)
                        else:
                            content_accum.append(chunk.content)
                    total_tokens += 1
                return ''.join(content_accum), ''.join(reasoning_accum), total_tokens, finish_reason

            loop = asyncio.get_running_loop()
            content, reasoning, total_generated, finish_reason = await loop.run_in_executor(None, _blocking_generate)

        message_payload = {
            'role': 'assistant',
            'content': content
        }
        if reasoning:
            message_payload['reasoning_content'] = reasoning

        return JSONResponse(
            content={
                'id': request_id,
                'object': 'chat.completion',
                'created': created_time,
                'model': model_id,
                'choices': [
                    {
                        'index': 0,
                        'message': message_payload,
                        'logprobs': None,
                        'finish_reason': finish_reason
                    }
                ],
                'usage': {
                    'prompt_tokens': 0,  # Placeholders
                    'completion_tokens': total_generated,
                    'total_tokens': total_generated
                }
            }
        )

    def run(self, host: str = '0.0.0.0', port: int = 11305, **kwargs) -> None:
        '''
        Start the FastAPI server using uvicorn.

        Args:
            host (str): Server host. Defaults to '0.0.0.0'.
            port (int): Server port. Defaults to 11305.
            **kwargs: Extra parameters for uvicorn.run.
        '''
        uvicorn.run(self.app, host=host, port=port, **kwargs)
