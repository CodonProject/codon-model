'''
强制 JSON 采样（grammar / constrained decoding）。

思路：把「合法 JSON」写成一个字符级增量状态机 `JSONGrammar`，再用 `TokenVocab`
（tokenizer 词表 + 预分类索引）在每一步算出「哪些 token 走完仍然落在合法 JSON 前缀上」，
由 `Sampler` 把这些 token 以外的 logits 全部置为 -inf。于是就算 logits 是随机的，
输出也一定可以被 `json.loads` 解析；给定 JSON Schema 时还会进一步约束键名、必填项、
类型与 enum 取值。

代价控制：词表被预分成「纯文字 token」（在字符串/自由键名状态下无条件合法）与
「含 `"` / `\\` / 控制字符的 token」（只校验特殊后缀），因此长字符串这种最常见的
情形是 O(1) 判定的；其余状态按首字符分桶，只对可能通过的那几桶做完整校验。

    from codon.model.grammar import build_json_constraint

    # 只保证是合法 JSON 对象
    c = build_json_constraint(tokenizer, mode='object', eos_token_id=im_end_id)

    # 保证是符合 schema 的 JSON
    c = build_json_constraint(tokenizer, schema={'type': 'object', 'properties': {...}},
                              eos_token_id=im_end_id)

    model.generate(ids, constraint=c, eos_token_id=im_end_id)          # 模型层
    for chunk in chat(..., response_format={'type': 'json_object'}):   # utils/generate + service
        ...

## 约束范围（JSON Schema 子集）

支持：
    type（object / array / string / number / integer / boolean / null）、properties、
    required、additionalProperties（bool 或子 schema）、enum、const、items、
    minItems / maxItems、minLength / maxLength，以及任意深度嵌套。
忽略（纯注解，不影响取值）：
    title / description / default / examples / $schema / $id / $comment / definitions /
    $defs / deprecated / readOnly / writeOnly / format / contentMediaType / contentEncoding。
显式报错（`UnsupportedSchemaError`，绝不静默降级）：
    anyOf / oneOf / allOf / not / $ref / pattern / patternProperties / propertyNames /
    if-then-else / minimum / maximum / exclusiveMinimum / exclusiveMaximum / multipleOf /
    uniqueItems / minProperties / maxProperties / 联合类型（type 数组）/ items 列表。

## 语义细节

* 只有在当前值「已经完整」时才允许结束：`eos` 在 JSON 完整之前被屏蔽，完整之后被强制，
  因此不会出现被 max_new_tokens 截断的半截 JSON（除非预算真的不够）。
* 字符串里拒绝裸控制字符（< 0x20），与 `json.loads` 的严格模式一致。
* `minLength` / `maxLength` 按解码后的字符数计算（一个转义序列算 1 个字符）。
* enum / const 只接受 `json.dumps(..., ensure_ascii=False)` 生成的规范文本
  （例如对象会带 `", "` / `": "` 分隔符）。
* 已经完整的文档只允许再输出空白字符。
'''

from codon import *
import json
import warnings


__all__ = [
    'UnsupportedSchemaError',
    'JSONConstraintError',
    'JSONSpec',
    'JSONGrammar',
    'TokenVocab',
    'JSONConstraint',
    'normalize_schema',
    'build_token_vocab',
    'build_json_constraint',
    'parse_response_format',
    'constraint_from_response_format',
]


class UnsupportedSchemaError(ValueError):
    '''JSON Schema 中出现了本实现不支持的关键字（不静默忽略，直接报错）。'''


class JSONConstraintError(RuntimeError):
    '''约束解码被喂进了当前语法不允许的 token（正常解码路径下不会发生）。'''


# ============================================================================
# JSON Schema（子集）校验与规范化
# ============================================================================

#: 纯注解关键字：对取值没有约束力，直接忽略。
_METADATA_KEYWORDS = frozenset({
    'title', 'description', 'default', 'examples', 'example', '$schema', '$id', '$comment',
    'definitions', '$defs', 'deprecated', 'readOnly', 'writeOnly', 'format',
    'contentMediaType', 'contentEncoding',
})

#: 本实现真正支持的约束关键字。
_SUPPORTED_KEYWORDS = frozenset({
    'type', 'properties', 'required', 'additionalProperties', 'enum', 'const',
    'items', 'minItems', 'maxItems', 'minLength', 'maxLength',
})

_TYPE_NAMES = frozenset({'object', 'array', 'string', 'number', 'integer', 'boolean', 'null'})

_MODES = ('json', 'object', 'array')


def _require_int(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{path} must be an integer, got {type(value).__name__}')
    if value < minimum:
        raise ValueError(f'{path} must be >= {minimum}, got {value}')
    return value


def _json_literal(value: Any) -> str:
    '''把 enum / const 的取值转成规范 JSON 文本（也是 FSM 唯一接受的写法）。'''
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'value {value!r} is not representable as JSON: {exc}') from exc


def normalize_schema(schema: Optional[Dict[str, Any]], path: str = '$') -> Optional[Dict[str, Any]]:
    '''校验并规范化 JSON Schema（本实现支持的子集）。

    不支持的约束关键字抛 `UnsupportedSchemaError`，类型/取值非法抛 `TypeError` /
    `ValueError`（例如「required 的键没写在 properties 里且 additionalProperties=false」
    这种永远无法满足的 schema）。
    '''
    if schema is None:
        return None
    if not isinstance(schema, dict):
        raise TypeError(f'{path} must be a dict (JSON Schema), got {type(schema).__name__}')

    unsupported = sorted(
        key for key in schema
        if key not in _SUPPORTED_KEYWORDS and key not in _METADATA_KEYWORDS
    )
    if unsupported:
        raise UnsupportedSchemaError(
            f'{path}: unsupported JSON Schema keyword(s) {unsupported}; '
            f'supported: {sorted(_SUPPORTED_KEYWORDS)}'
        )

    out: Dict[str, Any] = {}

    type_name = schema.get('type')
    if type_name is not None:
        if isinstance(type_name, (list, tuple, set)):
            raise UnsupportedSchemaError(
                f'{path}.type: union types (type arrays) are not supported; '
                f'use a single type, or enum/const'
            )
        if type_name not in _TYPE_NAMES:
            raise ValueError(
                f'{path}.type: unknown type {type_name!r}; expected one of {sorted(_TYPE_NAMES)}'
            )
        out['type'] = type_name

    if 'const' in schema and 'enum' in schema:
        if schema['const'] not in list(schema['enum']):
            raise ValueError(
                f'{path}: const {schema["const"]!r} is not in enum {schema["enum"]!r} (unsatisfiable)'
            )
    if 'enum' in schema:
        values = schema['enum']
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f'{path}.enum must be a non-empty list')
        out['enum'] = list(values)
    if 'const' in schema:
        out['const'] = schema['const']
    for value in ([out['const']] if 'const' in out else out.get('enum', [])):
        _json_literal(value)          # 提前暴露不可序列化的取值

    if 'properties' in schema:
        props = schema['properties']
        if not isinstance(props, dict):
            raise TypeError(f'{path}.properties must be a dict, got {type(props).__name__}')
        out['properties'] = {
            str(key): normalize_schema(value, f'{path}.properties[{key!r}]')
            for key, value in props.items()
        }

    if 'required' in schema:
        required = schema['required']
        if not isinstance(required, (list, tuple)) or any(not isinstance(k, str) for k in required):
            raise TypeError(f'{path}.required must be a list of strings')
        out['required'] = list(required)

    allow_extra = True
    if 'additionalProperties' in schema:
        extra = schema['additionalProperties']
        if isinstance(extra, bool):
            out['additionalProperties'] = extra
            allow_extra = extra
        elif isinstance(extra, dict):
            out['additionalProperties'] = normalize_schema(extra, f'{path}.additionalProperties')
        else:
            raise TypeError(f'{path}.additionalProperties must be a bool or a schema')

    if allow_extra is False and out.get('required') and 'properties' in out:
        missing = [key for key in out['required'] if key not in out['properties']]
        if missing:
            raise ValueError(
                f'{path}: required key(s) {missing} are not declared in properties while '
                f'additionalProperties is false (unsatisfiable)'
            )

    if 'items' in schema:
        items = schema['items']
        if isinstance(items, (list, tuple)):
            raise UnsupportedSchemaError(
                f'{path}.items: tuple validation (items as a list) is not supported; '
                f'use a single item schema'
            )
        out['items'] = normalize_schema(items, f'{path}.items')

    for key in ('minItems', 'maxItems', 'minLength', 'maxLength'):
        if key in schema:
            out[key] = _require_int(schema[key], f'{path}.{key}')

    if 'minItems' in out and 'maxItems' in out and out['minItems'] > out['maxItems']:
        raise ValueError(f'{path}: minItems > maxItems (unsatisfiable)')
    if 'minLength' in out and 'maxLength' in out and out['minLength'] > out['maxLength']:
        raise ValueError(f'{path}: minLength > maxLength (unsatisfiable)')

    return out


def _literal_candidates(schema: Dict[str, Any]) -> Optional[Tuple[str, ...]]:
    '''enum / const 对应的规范 JSON 文本集合（None 表示没有 enum/const 约束）。'''
    if 'const' in schema:
        return (_json_literal(schema['const']),)
    if 'enum' in schema:
        return tuple(dict.fromkeys(_json_literal(value) for value in schema['enum']))
    return None


def _resolve_schema(
    schema: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    '''把 (schema, mode) 合成为一个规范化后的根 schema。'''
    if mode is not None and mode not in _MODES:
        raise ValueError(f'unknown JSON mode {mode!r}; expected one of {_MODES}')
    if schema is None:
        return None if mode in (None, 'json') else {'type': mode}

    normalized = normalize_schema(schema)
    if mode in (None, 'json'):
        return normalized
    declared = normalized.get('type')
    if declared is not None and declared != mode:
        raise ValueError(f'JSON mode {mode!r} conflicts with schema type {declared!r}')
    out = dict(normalized)
    out['type'] = mode
    return out


# ============================================================================
# 字符级 JSON 状态机
# ============================================================================

_WS = frozenset(' \t\n\r')
_DIGITS = frozenset('0123456789')
_HEX = frozenset('0123456789abcdefABCDEF')
_ESCAPES = frozenset('"\\/bfnrt')

#: JSON 简单转义 -> 解码后的字符（键名匹配按解码结果进行）。
_ESCAPE_CHARS = {'"': '"', '\\': '\\', '/': '/', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}

# _feed 的返回值：ok = 吃掉这个字符；retry = 状态已切换，重新喂同一个字符；
# complete = 当前值刚好结束（字符不属于它），先收尾再重喂；fail = 非法。
_OK, _RETRY, _COMPLETE, _FAIL = 'ok', 'retry', 'complete', 'fail'

_ST_VALUE = 'value'
_ST_DONE = 'done'
_ST_OBJ_KEY_OR_END = 'obj_key_or_end'
_ST_OBJ_KEY = 'obj_key'                    # 逗号之后：只允许键，不允许 '}'
_ST_KEY = 'key'
_ST_KEY_ESC = 'key_esc'
_ST_KEY_U = 'key_u'
_ST_OBJ_COLON = 'obj_colon'
_ST_OBJ_VALUE = 'obj_value'
_ST_OBJ_COMMA_OR_END = 'obj_comma_or_end'
_ST_ARR_VALUE_OR_END = 'arr_value_or_end'
_ST_ARR_VALUE = 'arr_value'                # 逗号之后：只允许值，不允许 ']'
_ST_ARR_COMMA_OR_END = 'arr_comma_or_end'
_ST_STR = 'str'
_ST_STR_ESC = 'str_esc'
_ST_STR_U = 'str_u'
_ST_ENUM = 'enum'
_ST_LIT = 'lit'
_ST_NUM_SIGN = 'num_sign'
_ST_NUM_ZERO = 'num_zero'
_ST_NUM_INT = 'num_int'
_ST_NUM_DOT = 'num_dot'
_ST_NUM_FRAC = 'num_frac'
_ST_NUM_EXP = 'num_exp'
_ST_NUM_EXP_SIGN = 'num_exp_sign'
_ST_NUM_EXP_DIGITS = 'num_exp_digits'

#: 这些数字状态下数字已经是完整值，可以直接结束。
_TERMINABLE_NUM = frozenset({_ST_NUM_ZERO, _ST_NUM_INT, _ST_NUM_FRAC, _ST_NUM_EXP_DIGITS})

#: 值开始位置（等待一个新值）。
_VALUE_SLOTS = frozenset({_ST_VALUE, _ST_OBJ_VALUE, _ST_ARR_VALUE_OR_END, _ST_ARR_VALUE})


class JSONGrammar:
    '''增量式 JSON 字符状态机（可选 JSON Schema 约束）。

    典型用法（由 `JSONConstraint` 内部驱动，一般不需要直接用）::

        g = JSONGrammar({'type': 'object'})
        g.step('{')                 # True
        g.feed_text('"a": 1')       # True
        g.step('}')                 # True
        g.is_complete               # True
        g.document                  # '{"a": 1}'

    `step` 返回 False 表示该字符会让文本不再是「某个合法 JSON 文档的前缀」，
    此时状态会回滚到进入该字符之前（可以换一个字符继续）。
    `is_complete` 为 True 表示当前文本本身就是一个完整的合法 JSON 文档。
    '''

    def __init__(self, schema: Optional[Dict[str, Any]] = None, normalized: bool = False) -> None:
        self.root_schema: Optional[Dict[str, Any]] = (
            schema if normalized else _resolve_schema(schema)
        )
        self.reset()

    # ---- 状态 ----
    def reset(self) -> 'JSONGrammar':
        '''回到文档开头。'''
        self.state: str = _ST_VALUE
        self.stack: List[Tuple] = []
        self.text: Optional[List[str]] = []
        self.failed: bool = False

        self.slot: str = 'top'                  # 'top' / 'obj' / 'arr'：当前值所在槽位
        self.schema: Optional[Dict[str, Any]] = self.root_schema   # 当前值的 schema
        self.pending_key: Optional[str] = None  # 对象里正在取值的那把键

        self.key_decoded: str = ''                # 键名「解码后」的文本（\uXXXX 等按解码结果参与匹配）
        self.key_candidates: Optional[Tuple[str, ...]] = None
        self.hex_digits: str = ''

        self.str_len: int = 0
        self.str_min: int = 0
        self.str_max: Optional[int] = None

        self.hex_left: int = 0
        self.num_frac_ok: bool = True
        self.num_exp_ok: bool = True

        self.lit_target: str = ''
        self.lit_index: int = 0

        self.enum_candidates: Tuple[str, ...] = ()
        self.enum_viable: Tuple[int, ...] = ()
        self.enum_pos: int = 0
        return self

    def clone(self, probe: bool = False) -> 'JSONGrammar':
        '''复制当前状态；`probe=True` 时不记录文本（只用于试探某个 token 是否合法）。'''
        grammar = object.__new__(JSONGrammar)
        grammar.__dict__.update(self.__dict__)
        grammar.stack = list(self.stack)
        grammar.text = None if probe else list(self.text)
        return grammar

    # ---- 查询 ----
    @property
    def document(self) -> str:
        '''当前已生成的文本（probe 状态下为空串）。'''
        return '' if self.text is None else ''.join(self.text)

    @property
    def is_complete(self) -> bool:
        '''当前文本是否已经是一个完整的合法 JSON 文档。'''
        if self.failed:
            return False
        if self.state == _ST_DONE:
            return True
        if self.stack:
            return False                     # 还有没闭合的容器
        if self.state in _TERMINABLE_NUM:
            return True                      # 顶层数字可以就地结束
        if self.state == _ST_ENUM and self._enum_complete():
            return True
        return False

    def absorbs_plain_text(self) -> bool:
        '''当前状态是否「吃掉任意纯文字字符后仍停留在同一状态」。

        只有两种状态成立：没有长度限制的字符串内容、键名自由的键内容。
        这是词表快速筛选的前提：此时纯文字 token 一定合法，只需真正校验含
        `"` / `\\` / 控制字符的 token（且可以跳过它们前面的纯文字部分）。
        '''
        if self.state == _ST_STR:
            return self.str_min == 0 and self.str_max is None
        if self.state == _ST_KEY:
            return self.key_candidates is None
        return False

    # ---- 推进 ----
    def step(self, ch: str) -> bool:
        '''吃掉一个字符；返回 False 表示它破坏了 JSON 前缀（此时状态回滚，可换字符重试）。'''
        if self.failed:
            return False

        snapshot = self.__dict__.copy()
        snapshot['stack'] = list(self.stack)
        snapshot['text'] = None if self.text is None else list(self.text)

        for _ in range(8):
            action = self._feed(ch)
            if action == _OK:
                if self.text is not None:
                    self.text.append(ch)
                return True
            if action == _COMPLETE:
                # 字符不属于正在结束的那个值（例如数字后面的 '}'）：先收尾再重喂
                self._value_done()
                continue
            if action == _RETRY:
                continue
            break

        self.__dict__.update(snapshot)
        return False

    def feed_text(self, text: str) -> bool:
        '''依次吃掉一串字符；任意一个字符非法即返回 False。'''
        for ch in text:
            if not self.step(ch):
                return False
        return True

    # ---- 各状态的转移 ----
    def _feed(self, ch: str) -> str:
        state = self.state

        if state in _VALUE_SLOTS:
            if state == _ST_ARR_VALUE_OR_END and ch == ']':
                frame = self.stack[-1]
                if frame[3] < frame[4]:      # count < minItems
                    return _FAIL
                return self._close('arr')
            return self._start_value(ch)

        if state == _ST_DONE:
            return _OK if ch in _WS else _FAIL

        if state == _ST_OBJ_KEY_OR_END or state == _ST_OBJ_KEY:
            if ch in _WS:
                return _OK
            if ch == '"':
                return self._begin_key()
            if ch == '}' and state == _ST_OBJ_KEY_OR_END:
                return self._close('obj') if self._required_ok() else _FAIL
            return _FAIL

        if state == _ST_KEY:
            return self._feed_key(ch)
        if state in (_ST_KEY_ESC, _ST_STR_ESC):
            return self._feed_escape(ch)
        if state in (_ST_KEY_U, _ST_STR_U):
            return self._feed_unicode(ch)
        if state == _ST_STR:
            return self._feed_string(ch)

        if state == _ST_OBJ_COLON:
            if ch in _WS:
                return _OK
            if ch != ':':
                return _FAIL
            frame = self.stack[-1]
            props, extra = frame[2], frame[5]
            self.schema = props.get(self.pending_key, extra) if props else extra
            self.slot = 'obj'
            self.state = _ST_OBJ_VALUE
            return _OK

        if state == _ST_OBJ_COMMA_OR_END:
            if ch in _WS:
                return _OK
            if ch == ',':
                frame = self.stack[-1]
                if not frame[6] and not any(key not in frame[4] for key in frame[2]):
                    return _FAIL                # 已经没有可写的键名了，只能 '}'
                self.state = _ST_OBJ_KEY          # 逗号之后必须还有键，避免出现 {"a":1,}
                return _OK
            if ch == '}':
                return self._close('obj') if self._required_ok() else _FAIL
            return _FAIL

        if state == _ST_ARR_COMMA_OR_END:
            if ch in _WS:
                return _OK
            frame = self.stack[-1]
            if ch == ',':
                if frame[5] is not None and frame[3] >= frame[5]:
                    return _FAIL                # 已到 maxItems，只能 ']'
                self.schema = frame[2]
                self.slot = 'arr'
                self.state = _ST_ARR_VALUE        # 逗号之后必须还有元素，避免出现 [1,]
                return _OK
            if ch == ']':
                if frame[3] < frame[4]:
                    return _FAIL
                return self._close('arr')
            return _FAIL

        if state == _ST_LIT:
            return self._feed_literal(ch)
        if state == _ST_ENUM:
            return self._feed_enum(ch)
        if state.startswith('num_'):
            return self._feed_number(ch)
        return _FAIL

    def _start_value(self, ch: str) -> str:
        '''值开始位置：按 schema 决定允许的开头。'''
        if ch in _WS:
            return _OK

        if self.slot == 'arr' and self.stack:
            frame = self.stack[-1]
            max_items = frame[5]
            if max_items is not None and frame[3] >= max_items:
                return _FAIL

        schema = self.schema or {}
        candidates = _literal_candidates(schema)
        if candidates is not None:
            self.enum_candidates = candidates
            self.enum_viable = tuple(range(len(candidates)))
            self.enum_pos = 0
            self.state = _ST_ENUM
            return _RETRY

        type_name = schema.get('type')

        if ch == '{':
            if type_name not in (None, 'object'):
                return _FAIL
            self._open_object()
            return _OK
        if ch == '[':
            if type_name not in (None, 'array'):
                return _FAIL
            self._open_array()
            return _OK
        if ch == '"':
            if type_name not in (None, 'string'):
                return _FAIL
            self._begin_string()
            return _OK
        if ch in _DIGITS or ch == '-':
            if type_name not in (None, 'number', 'integer'):
                return _FAIL
            integer = type_name == 'integer'
            self.num_frac_ok = not integer
            self.num_exp_ok = not integer
            if ch == '-':
                self.state = _ST_NUM_SIGN
            elif ch == '0':
                self.state = _ST_NUM_ZERO
            else:
                self.state = _ST_NUM_INT
            return _OK
        if ch in ('t', 'f'):
            if type_name not in (None, 'boolean'):
                return _FAIL
            self.lit_target = 'true' if ch == 't' else 'false'
            self.lit_index = 1
            self.state = _ST_LIT
            return _OK
        if ch == 'n':
            if type_name not in (None, 'null'):
                return _FAIL
            self.lit_target = 'null'
            self.lit_index = 1
            self.state = _ST_LIT
            return _OK
        return _FAIL

    def _open_object(self) -> None:
        frame_slot = self.slot
        schema = self.schema or {}
        props = schema.get('properties') or {}
        required = frozenset(schema.get('required') or ())
        extra = schema.get('additionalProperties', True)
        if extra is False:
            extra_schema, allow_extra = None, False
        elif isinstance(extra, dict):
            extra_schema, allow_extra = extra, True
        else:
            extra_schema, allow_extra = None, True
        self.stack.append(('obj', frame_slot, props, required, frozenset(), extra_schema, allow_extra))
        self.state = _ST_OBJ_KEY_OR_END

    def _open_array(self) -> None:
        frame_slot = self.slot
        schema = self.schema or {}
        item_schema = schema.get('items')
        self.stack.append((
            'arr', frame_slot, item_schema,
            0, schema.get('minItems') or 0, schema.get('maxItems'),
        ))
        self.schema = item_schema
        self.slot = 'arr'
        self.state = _ST_ARR_VALUE_OR_END

    def _required_ok(self) -> bool:
        frame = self.stack[-1]
        return frame[3] <= frame[4]

    def _close(self, kind: str) -> str:
        '''闭合最内层容器，并把它当作一个「已完成的值」交回外层。'''
        if not self.stack:
            return _FAIL
        frame = self.stack.pop()
        if kind == 'obj' and not (frame[3] <= frame[4]):
            self.stack.append(frame)
            return _FAIL
        self.slot = frame[1]
        self._value_done()
        return _OK

    def _value_done(self) -> None:
        '''当前值结束：按它所在的槽位回到外层的结构状态。'''
        slot = self.slot
        self.pending_key = None
        self.schema = None
        if slot == 'top':
            self.state = _ST_DONE
        elif slot == 'obj':
            self.state = _ST_OBJ_COMMA_OR_END
        elif slot == 'arr':
            kind, outer_slot, item_schema, count, min_items, max_items = self.stack[-1]
            self.stack[-1] = (kind, outer_slot, item_schema, count + 1, min_items, max_items)
            self.state = _ST_ARR_COMMA_OR_END
        else:
            self.failed = True

    def _begin_string(self) -> None:
        schema = self.schema or {}
        self.str_len = 0
        self.str_min = int(schema.get('minLength') or 0)
        self.str_max = schema.get('maxLength')
        self.state = _ST_STR

    def _add_str_char(self) -> bool:
        '''字符串里又吃掉一个「字符」（转义序列整体算一个）。'''
        if self.str_max is not None and self.str_len >= self.str_max:
            return False
        self.str_len += 1
        return True

    def _feed_string(self, ch: str) -> str:
        if ch == '"':
            if self.str_len < self.str_min:
                return _FAIL
            self._value_done()
            return _OK
        if ch == '\\':
            if self.str_max is not None and self.str_len >= self.str_max:
                return _FAIL
            self.state = _ST_STR_ESC
            return _OK
        if ord(ch) < 0x20:
            return _FAIL
        if not self._add_str_char():
            return _FAIL
        return _OK

    def _begin_key(self) -> str:
        frame = self.stack[-1]
        if frame[6]:                          # allow_extra：键名自由
            self.key_candidates = None
        else:
            props, seen = frame[2], frame[4]
            candidates = tuple(key for key in props if key not in seen)
            if not candidates:
                return _FAIL                    # 没有可用键名了，只能 '}'
            self.key_candidates = candidates
        self.key_decoded = ''
        self.state = _ST_KEY
        return _OK

    def _push_key_char(self, ch: str) -> str:
        '''把一个「解码后的键名字符」并入当前键名，并按候选键名做前缀剪枝。'''
        if self.key_candidates is not None:
            prefix = self.key_decoded + ch
            if not any(name.startswith(prefix) for name in self.key_candidates):
                return _FAIL
        self.key_decoded += ch
        return _OK

    def _feed_key(self, ch: str) -> str:
        if ch == '"':
            if self.key_candidates is not None and self.key_decoded not in self.key_candidates:
                return _FAIL
            return self._end_key()
        if ch == '\\':
            if self.key_candidates is not None:
                # 严格对象的键名必须字面量书写：转义序列会让「原始文本 -> 键名」不再可控，
                # 而且中途无法判断它是否还能落到某个候选键名上（会走进死胡同）。
                return _FAIL
            self.state = _ST_KEY_ESC
            return _OK
        if ord(ch) < 0x20:
            return _FAIL
        return self._push_key_char(ch)

    def _end_key(self) -> str:
        key = self.key_decoded
        frame = self.stack[-1]
        props, required, seen, extra, allow_extra = frame[2], frame[3], frame[4], frame[5], frame[6]
        if not allow_extra and key not in props:
            return _FAIL
        self.stack[-1] = ('obj', frame[1], props, required, seen | {key}, extra, allow_extra)
        self.pending_key = key
        self.state = _ST_OBJ_COLON
        return _OK

    def _feed_escape(self, ch: str) -> str:
        in_key = self.state == _ST_KEY_ESC
        if ch in _ESCAPES:
            if in_key:
                self.state = _ST_KEY
                return self._push_key_char(_ESCAPE_CHARS[ch])
            self.state = _ST_STR
            return _OK if self._add_str_char() else _FAIL
        if ch == 'u':
            self.hex_left = 4
            self.hex_digits = ''
            self.state = _ST_KEY_U if in_key else _ST_STR_U
            return _OK
        return _FAIL

    def _feed_unicode(self, ch: str) -> str:
        if ch not in _HEX:
            return _FAIL
        self.hex_digits += ch
        self.hex_left -= 1
        if self.hex_left > 0:
            return _OK
        if self.state == _ST_KEY_U:
            self.state = _ST_KEY
            return self._push_key_char(chr(int(self.hex_digits, 16)))
        self.state = _ST_STR
        return _OK if self._add_str_char() else _FAIL

    def _feed_literal(self, ch: str) -> str:
        target = self.lit_target
        if self.lit_index < len(target) and ch == target[self.lit_index]:
            self.lit_index += 1
            if self.lit_index == len(target):
                self._value_done()
            return _OK
        return _COMPLETE if self.lit_index == len(target) else _FAIL

    def _enum_complete(self) -> bool:
        return any(len(self.enum_candidates[i]) == self.enum_pos for i in self.enum_viable)

    def _feed_enum(self, ch: str) -> str:
        pos = self.enum_pos
        nxt = tuple(
            i for i in self.enum_viable
            if len(self.enum_candidates[i]) > pos and self.enum_candidates[i][pos] == ch
        )
        if nxt:
            self.enum_viable = nxt
            self.enum_pos = pos + 1
            return _OK
        if self._enum_complete():
            return _COMPLETE                 # 由 step() 负责 _value_done()
        return _FAIL

    def _feed_number(self, ch: str) -> str:
        state = self.state
        if state == _ST_NUM_SIGN:
            if ch == '0':
                self.state = _ST_NUM_ZERO
                return _OK
            if ch in _DIGITS:
                self.state = _ST_NUM_INT
                return _OK
            return _FAIL
        if state == _ST_NUM_ZERO:
            if ch == '.' and self.num_frac_ok:
                self.state = _ST_NUM_DOT
                return _OK
            if ch in 'eE' and self.num_exp_ok:
                self.state = _ST_NUM_EXP
                return _OK
            return _COMPLETE
        if state == _ST_NUM_INT:
            if ch in _DIGITS:
                return _OK
            if ch == '.' and self.num_frac_ok:
                self.state = _ST_NUM_DOT
                return _OK
            if ch in 'eE' and self.num_exp_ok:
                self.state = _ST_NUM_EXP
                return _OK
            return _COMPLETE
        if state == _ST_NUM_DOT:
            if ch in _DIGITS:
                self.state = _ST_NUM_FRAC
                return _OK
            return _FAIL
        if state == _ST_NUM_FRAC:
            if ch in _DIGITS:
                return _OK
            if ch in 'eE' and self.num_exp_ok:
                self.state = _ST_NUM_EXP
                return _OK
            return _COMPLETE
        if state == _ST_NUM_EXP:
            if ch in '+-':
                self.state = _ST_NUM_EXP_SIGN
                return _OK
            if ch in _DIGITS:
                self.state = _ST_NUM_EXP_DIGITS
                return _OK
            return _FAIL
        if state == _ST_NUM_EXP_SIGN:
            if ch in _DIGITS:
                self.state = _ST_NUM_EXP_DIGITS
                return _OK
            return _FAIL
        if state == _ST_NUM_EXP_DIGITS:
            return _OK if ch in _DIGITS else _COMPLETE
        return _FAIL


# ============================================================================
# 词表索引
# ============================================================================

def _is_plain(ch: str) -> bool:
    '''「纯文字」字符：JSON 字符串里可以随便出现的字符。'''
    return ord(ch) >= 0x20 and ch != '"' and ch != '\\'


def _restricted_index(text: str) -> int:
    '''文本里第一个受限字符（`"` / `\\` / 控制字符）的下标；全是纯文字时返回 -1。'''
    for index, ch in enumerate(text):
        if not _is_plain(ch):
            return index
    return -1


class TokenVocab:
    '''tokenizer 词表的「token -> 文本」表 + 逐 token 校验所需的索引。

    只保留能安全参与文本拼接的 token：特殊 token、safe_escape、解码出替换字符
    （`\\ufffd`，说明该 token 是多字节字符的碎片）以及空文本 token 都会被剔除。

    为了让「每一步筛出合法 token」足够快，这里预先把词表分成两类：

    * 纯文字 token（所有字符都不是 `"` / `\\` / 控制字符）：在「吸收纯文字」的语法状态下
      一定合法，无需校验；
    * 其余 token：记下第一个受限字符的位置，在前一种状态下只需校验它的后缀。

    其它状态下按首字符分桶校验，配合首字符探测把候选量压到很小。
    '''

    def __init__(self, texts: Dict[int, str], vocab_size: Optional[int] = None) -> None:
        clean: Dict[int, str] = {}
        for token_id, text in texts.items():
            token_id = int(token_id)
            if not text or '\ufffd' in text:
                continue
            clean[token_id] = text

        self.texts = clean
        self.vocab_size = int(vocab_size) if vocab_size else (max(clean) + 1 if clean else 0)
        self.all_ids: Tuple[int, ...] = tuple(sorted(clean))

        plain: List[int] = []
        restricted: List[Tuple[int, int]] = []
        by_first_char: Dict[str, List[int]] = {}
        for token_id in self.all_ids:
            text = clean[token_id]
            by_first_char.setdefault(text[0], []).append(token_id)
            start = _restricted_index(text)
            if start < 0:
                plain.append(token_id)
            else:
                restricted.append((token_id, start))

        #: 纯文字 token（可被「吸收纯文字」的状态无条件接受）。
        self.plain_ids: Tuple[int, ...] = tuple(plain)
        #: (token_id, 首个受限字符下标)：校验时可以从该下标开始喂给语法。
        self.restricted: Tuple[Tuple[int, int], ...] = tuple(restricted)
        self.first_chars: Tuple[str, ...] = tuple(by_first_char)
        self.by_first_char: Dict[str, Tuple[int, ...]] = {
            ch: tuple(ids) for ch, ids in by_first_char.items()
        }

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> 'TokenVocab':
        '''从 `PackedTokenizer`（或任何提供 vocab_size / decode / token_to_id 的对象）建表。'''
        size = int(tokenizer.vocab_size)
        banned = set()

        fast = getattr(tokenizer, 'fast_tokenizer', None)
        if fast is not None:
            for token_id in (getattr(fast, 'all_special_ids', None) or []):
                banned.add(int(token_id))
            try:
                banned.update(int(token_id) for token_id in fast.get_added_vocab().values())
            except Exception:
                pass

        escape_name = getattr(tokenizer, 'safe_escape', None)
        if escape_name:
            escape_id = None
            try:
                escape_id = tokenizer.token_to_id(escape_name)
            except Exception:
                escape_id = None
            if escape_id is not None:
                banned.add(int(escape_id))
        escape_id = getattr(tokenizer, 'safe_escape_id', None)
        if escape_id is not None:
            banned.add(int(escape_id))

        texts: Dict[int, str] = {}
        for token_id in range(size):
            if token_id in banned:
                continue
            try:
                text = tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                continue
            if not text or '\ufffd' in text:
                continue
            texts[token_id] = text
        return cls(texts, vocab_size=size)

    def allowed_ids(self, grammar: JSONGrammar) -> List[int]:
        '''当前语法状态下所有「走完仍是合法 JSON 前缀」的 token id。'''
        if grammar.failed:
            return []

        if grammar.absorbs_plain_text():
            out = list(self.plain_ids)
            for token_id, start in self.restricted:
                probe = grammar.clone(probe=True)
                if probe.feed_text(self.texts[token_id][start:]):
                    out.append(token_id)
            return out

        out: List[int] = []
        for ch in self.first_chars:
            probe = grammar.clone(probe=True)
            if not probe.step(ch):
                continue                       # 该首字符不被接受：整桶跳过
            for token_id in self.by_first_char[ch]:
                text = self.texts[token_id]
                if len(text) == 1:
                    out.append(token_id)
                    continue
                branch = grammar.clone(probe=True)
                if branch.feed_text(text):
                    out.append(token_id)
        return out


# ============================================================================
# 约束对象
# ============================================================================

class JSONConstraint:
    '''把 `JSONGrammar` 与 `TokenVocab` 绑定，供 `Sampler` 逐步调用。

    每个 batch 行维护一份独立的语法状态。接口：

        mask_logits(logits)    -> 把不允许的 token 置为 -inf 后返回
        apply_forcing(tokens)  -> 文档已完整时把该行 token 强制换成 eos
        advance(tokens)        -> 记录本步采样的 token，推进语法状态
    '''

    def __init__(
        self,
        vocab: TokenVocab,
        schema: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        eos_token_id: Optional[int] = None,
        batch_size: int = 1,
    ) -> None:
        self.vocab = vocab
        self.root_schema = _resolve_schema(schema, mode)
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)
        self.warned_empty = False
        self._grammars: List[JSONGrammar] = []
        self.reset(batch_size)

    # ---- 状态 ----
    def reset(self, batch_size: Optional[int] = None) -> 'JSONConstraint':
        '''重置语法状态（每个 batch 行一份）。'''
        size = int(batch_size) if batch_size else (len(self._grammars) or 1)
        self._grammars = [JSONGrammar(self.root_schema, normalized=True) for _ in range(size)]
        return self

    def _sync(self, batch: int) -> None:
        if len(self._grammars) != batch:
            self.reset(batch)

    @property
    def grammars(self) -> List[JSONGrammar]:
        return list(self._grammars)

    @property
    def is_complete(self) -> bool:
        '''所有行是否都已经生成出完整 JSON。'''
        return all(grammar.is_complete for grammar in self._grammars)

    def document(self, row: int = 0) -> str:
        '''第 row 行已生成的 JSON 文本。'''
        return self._grammars[row].document

    def allowed_token_ids(self, row: int = 0) -> List[int]:
        '''第 row 行当前允许的 token id（文档已完整时只剩空白字符 token）。'''
        return self.vocab.allowed_ids(self._grammars[row])

    # ---- 采样钩子 ----
    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        '''把不合法 token 的 logits 置为 -inf（不修改入参）。'''
        self._sync(logits.shape[0])
        mask = torch.full(logits.shape, float('-inf'), dtype=logits.dtype, device=logits.device)

        for row, grammar in enumerate(self._grammars):
            if grammar.failed:
                mask[row].fill_(0.0)          # 状态已坏：不再约束该行（advance 会先抛错）
                continue

            ids = self.allowed_token_ids(row)
            if not ids:
                if not self.warned_empty:
                    self.warned_empty = True
                    warnings.warn(
                        'JSON constraint found no legal next token in the vocabulary; '
                        'falling back to unconstrained sampling for that step',
                        RuntimeWarning,
                    )
                mask[row].fill_(0.0)
                continue

            index = torch.tensor(ids, dtype=torch.long, device=logits.device)
            index = index[index < logits.shape[-1]]
            if index.numel() == 0:
                # 词表和模型输出维度对不上（或全部越界）：不约束这一行，避免全 -inf
                mask[row].fill_(0.0)
                continue
            mask[row, index] = 0.0

        return logits + mask

    def apply_forcing(self, next_token: torch.Tensor) -> torch.Tensor:
        '''文档已完整的行，强制输出 eos（没有配 eos 时保持原样）。'''
        if self.eos_token_id is None:
            return next_token
        self._sync(next_token.shape[0])
        for row, grammar in enumerate(self._grammars):
            if grammar.is_complete:
                next_token[row] = self.eos_token_id
        return next_token

    def advance(self, next_token: torch.Tensor) -> None:
        '''记录本步采样的 token 并推进状态（eos / 非文本 token 直接忽略）。'''
        ids = next_token.detach().reshape(-1).tolist() if hasattr(next_token, 'detach') else list(next_token)
        self._sync(len(ids))
        for row, token_id in enumerate(ids):
            token_id = int(token_id)
            if self.eos_token_id is not None and token_id == self.eos_token_id:
                continue
            text = self.vocab.texts.get(token_id)
            if text is None:
                continue
            grammar = self._grammars[row]
            if not grammar.feed_text(text):
                raise JSONConstraintError(
                    f'row {row}: token {token_id} ({text!r}) is not allowed by the current JSON grammar'
                )

    # ---- 便捷调试 ----
    def commit_text(self, text: str, row: int = 0) -> None:
        '''手工把一段文本喂进某一行（测试/回放用）。'''
        if not self._grammars[row].feed_text(text):
            raise JSONConstraintError(f'row {row}: text {text!r} is not valid JSON prefix')


# ============================================================================
# 工厂 / response_format 解析
# ============================================================================

#: 词表缓存的属性名（挂在 tokenizer 实例上，避免每次请求重建索引）。
_VOCAB_CACHE_ATTR = '_codon_json_vocab'


def build_token_vocab(tokenizer: Any) -> TokenVocab:
    '''建（并缓存）tokenizer 的词表索引。'''
    cached = getattr(tokenizer, _VOCAB_CACHE_ATTR, None)
    if isinstance(cached, TokenVocab):
        return cached
    vocab = TokenVocab.from_tokenizer(tokenizer)
    try:
        setattr(tokenizer, _VOCAB_CACHE_ATTR, vocab)
    except Exception:
        pass
    return vocab


def build_json_constraint(
    tokenizer: Any,
    schema: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    eos_token_id: Optional[int] = None,
) -> JSONConstraint:
    '''
    构建强制 JSON 采样的约束对象。

    Args:
        tokenizer: `PackedTokenizer`（或等价对象），用于取词表。
        schema (dict, optional): JSON Schema（本实现支持的子集），None 表示只要求是合法 JSON。
        mode (str, optional): 'json' / 'object' / 'array'，限定顶层取值类型。
        eos_token_id (int, optional): JSON 完整后强制输出的结束 token（一般为 [im_end]）。

    Returns:
        JSONConstraint: 可直接传给 `Sampler(constraint=...)` 或 `model.generate(constraint=...)`。
    '''
    return JSONConstraint(
        build_token_vocab(tokenizer),
        schema=schema,
        mode=mode,
        eos_token_id=eos_token_id,
    )


@dataclass
class JSONSpec:
    '''
    response_format 解析结果。

    Attributes:
        schema (dict, optional): 规范化后的 JSON Schema；None 表示不约束结构。
        mode (str, optional): 'json' / 'object' / 'array'；None 表示由 schema 决定。
    '''
    schema: Optional[Dict[str, Any]] = None
    mode: Optional[str] = None


def parse_response_format(response_format: Any) -> Optional[JSONSpec]:
    '''
    解析 OpenAI 风格的 `response_format`；返回 None 表示不做 JSON 约束。

    支持 `{'type': 'text'}`、`{'type': 'json'}`、`{'type': 'json_object'}`、
    `{'type': 'json_array'}`、`{'type': 'json_schema', 'json_schema': {'schema': {...}}}`，
    以及直接给字符串（等价于 `{'type': ...}`）和 `{'type': 'json_object', 'schema': {...}}`。
    '''
    if response_format is None:
        return None
    if isinstance(response_format, str):
        response_format = {'type': response_format}
    if not isinstance(response_format, dict):
        raise TypeError(
            f'response_format must be a dict or str, got {type(response_format).__name__}'
        )

    kind = response_format.get('type', 'text')
    if kind in (None, 'text', 'none'):
        return None
    if kind == 'json':
        return JSONSpec(None, 'json')
    if kind in ('json_object', 'json_array'):
        mode = 'object' if kind == 'json_object' else 'array'
        schema = response_format.get('schema')
        if schema is None:
            return JSONSpec(None, mode)
        return JSONSpec(_resolve_schema(schema, mode), None)
    if kind == 'json_schema':
        payload = response_format.get('json_schema')
        if isinstance(payload, dict) and 'schema' in payload:
            schema = payload['schema']
        else:
            schema = payload
        if schema is None:
            schema = response_format.get('schema')
        if schema is None:
            raise ValueError("response_format.type='json_schema' requires a json_schema.schema")
        return JSONSpec(_resolve_schema(schema, None), None)

    raise ValueError(
        f'unsupported response_format type {kind!r}; expected one of '
        f"'text', 'json', 'json_object', 'json_array', 'json_schema'"
    )


def constraint_from_response_format(
    tokenizer: Any,
    response_format: Any,
    eos_token_id: Optional[int] = None,
) -> Optional[JSONConstraint]:
    '''`parse_response_format` + `build_json_constraint` 的组合；None 表示不约束。'''
    spec = parse_response_format(response_format)
    if spec is None:
        return None
    return build_json_constraint(
        tokenizer,
        schema=spec.schema,
        mode=spec.mode,
        eos_token_id=eos_token_id,
    )
