import os
import sys
import json
import re
import hashlib
import typing as _tp
from typing import Dict, Set, Any, Union, Callable, Optional

try:
    import yaml
except ImportError:
    yaml = None


# Annotation -> schema rule
def _ann_to_rule(ann: Any) -> Any:
    """把类型注解转成 BasicConfig schema 规则。

    注意：Union 分支保留完整的 get_args（含 NoneType），而不是剥掉 NoneType。
    因为 BasicConfig._is_optional_rule 靠 'type(None) in tuple' 识别 Optional 语义；
    _verify_rule 在做 isinstance 检查时会自己把 NoneType 剔除，所以保留它安全。
    """
    if ann is None or ann is type(None):
        return type(None)

    origin = _tp.get_origin(ann)

    if origin is _tp.Union:
        return _tp.get_args(ann)

    if origin is _tp.Literal:
        choices = _tp.get_args(ann)
        def _literal_check(v, _choices=set(choices)):
            return v in _choices
        _literal_check.__name__ = f'Literal[{sorted(map(str, choices))}]'
        return _literal_check

    if origin is not None:
        return None

    if isinstance(ann, type):
        return ann

    return None


def _copy_default(val: Any) -> Any:
    """嵌套 BasicConfig 默认值转 dict。

    默认值里放 Optim() 这样的实例时，如果不转 dict，这个实例会被所有
    Trainer() 共享；而 update 对已存在的嵌套节点走就地合并（.update()），
    一旦某个实例传了 dict 覆盖，就会污染共享默认。转成 dict 后每次构造
    都按 schema 新建独立节点。
    """
    if isinstance(val, BasicConfig):
        return val.to_dict()
    return val


# Field metadata (like dataclasses.field, for advanced cases)
class _FieldSpec:
    __slots__ = ('default', 'default_factory', 'validator', 'description')

    def __init__(self, default, default_factory, validator, description):
        self.default = default
        self.default_factory = default_factory
        self.validator = validator
        self.description = description


def field(
    *,
    default: Any = ...,
    default_factory: Callable[[], Any] = None,
    validator: Callable[[Any], bool] = None,
    description: str = '',
) -> Any:
    """
    Field metadata for @configclass. Use as the value of an annotated attribute.

    Example:
        @configclass
        class Cfg:
            temperature: float = field(
                default=1.0,
                validator=lambda x: 0 < x < 10,
                description='Sampling temperature',
            )
    """
    return _FieldSpec(default, default_factory, validator, description)


class BasicConfig:
    __schema__: Dict[str, Any] = {}
    _SECRET_KEYS: Set[str] = {'password', 'token', 'api_key', 'secret', 'access_token', 'private_key'}

    def __init__(self, **kwargs):
        super().__setattr__('_data', {})
        super().__setattr__('_frozen', False)
        super().__setattr__('_callbacks', {})
        self.update(kwargs)

    def _parse_string_value(self, val_str: str) -> Any:
        if not isinstance(val_str, str):
            return val_str
        s = val_str.strip()
        low = s.lower()
        if low == 'true':
            return True
        if low == 'false':
            return False
        if low == 'none' or s == '':
            return None
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return s

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self._data.items():
            if isinstance(value, BasicConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        current = self
        for k in keys:
            if isinstance(current, BasicConfig) and k in current._data:
                current = current._data[k]
            else:
                return default
        return self.resolve_value(current)

    def set_by_path(self, path: str, value: Any) -> None:
        self._check_frozen()
        keys = path.split('.')
        current = self
        for k in keys[:-1]:
            if not isinstance(current, BasicConfig):
                raise TypeError(f"cannot descend into '{k}': parent is {type(current).__name__}, not a config node")
            if k not in current._data:
                schema = getattr(current, '__schema__', {})
                target_cls = schema.get(k)
                if isinstance(target_cls, type) and issubclass(target_cls, BasicConfig):
                    current._data[k] = target_cls()
                else:
                    current._data[k] = BasicConfig()
            current = current._data[k]
        final_key = keys[-1]
        if not isinstance(current, BasicConfig):
            raise TypeError(f"cannot set '{final_key}': parent is {type(current).__name__}, not a config node")
        schema = getattr(current, '__schema__', {})
        target_cls = schema.get(final_key)
        if isinstance(value, dict):
            if isinstance(target_cls, type) and issubclass(target_cls, BasicConfig):
                current._data[final_key] = target_cls(**value)
            else:
                current._data[final_key] = BasicConfig(**value)
        else:
            current._data[final_key] = value
        current._trigger_callback(final_key, current._data[final_key])

    def update(self, other: Union[Dict[str, Any], 'BasicConfig']) -> None:
        self._check_frozen()
        if isinstance(other, BasicConfig):
            other = other.to_dict()
        schema = getattr(self, '__schema__', {})
        for key, value in other.items():
            if isinstance(value, dict):
                target_cls = schema.get(key)
                if isinstance(target_cls, type) and issubclass(target_cls, BasicConfig):
                    if key in self._data and isinstance(self._data[key], target_cls):
                        self._data[key].update(value)
                    else:
                        self._data[key] = target_cls(**value)
                elif key in self._data and isinstance(self._data[key], BasicConfig):
                    self._data[key].update(value)
                else:
                    self._data[key] = BasicConfig(**value)
            else:
                self._data[key] = value
                self._trigger_callback(key, value)

    def resolve_value(self, value: Any) -> Any:
        if isinstance(value, str) and "${" in value:
            pattern = re.compile(r"\$\{([^}]+)\}")
            matches = pattern.findall(value)
            for match in matches:
                if ":=" in match:
                    var_path, default_str = match.split(":=", 1)
                else:
                    var_path, default_str = match, None
                ref_val = None
                if var_path.startswith("env:"):
                    env_key = var_path[4:]
                    env_val = os.environ.get(env_key)
                    if env_val is not None:
                        ref_val = self._parse_string_value(env_val)
                else:
                    ref_val = self.get(var_path)
                    if ref_val is None:
                        env_val = os.environ.get(var_path)
                        if env_val is not None:
                            ref_val = self._parse_string_value(env_val)
                if ref_val is None and default_str is not None:
                    ref_val = self._parse_string_value(default_str)
                if ref_val is None:
                    raise ValueError(f"Referenced variable '${{{match}}}' not found in config or environment.")
                ref_val = self.resolve_value(ref_val)
                if value == f"${{{match}}}":
                    return ref_val
                value = value.replace(f"${{{match}}}", str(ref_val))
        return value

    def validate(self) -> None:
        schema = getattr(self, '__schema__', {})
        for path, expected_rule in schema.items():
            val = self.get(path)
            if val is None:
                if not self._is_optional_rule(expected_rule):
                    raise ValueError(f"[Schema Error] Required config key '{path}' is missing.")
                continue
            self._verify_rule(path, val, expected_rule)
        for k, v in self._data.items():
            if isinstance(v, BasicConfig):
                v.validate()

    def _is_optional_rule(self, rule: Any) -> bool:
        if rule is None or rule is type(None):
            return True
        if isinstance(rule, tuple) and type(None) in rule:
            return True
        return False

    def _verify_rule(self, path: str, val: Any, rule: Any) -> None:
        if isinstance(rule, type) and issubclass(rule, BasicConfig):
            if not isinstance(val, rule):
                raise TypeError(f"[Schema Error] '{path}' expected class {rule.__name__}, got {type(val).__name__}")
            return
        if isinstance(rule, type) or (isinstance(rule, tuple) and all(isinstance(r, type) for r in rule)):
            types = rule
            if isinstance(rule, tuple):
                types = tuple(r for r in rule if r is not type(None))
            if not isinstance(val, types):
                raise TypeError(f"[Schema Error] '{path}' expected {rule}, got {type(val).__name__} ({val})")
        elif callable(rule):
            if not rule(val):
                raise ValueError(f"[Schema Error] '{path}' value '{val}' failed validation function rule.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BasicConfig':
        return cls(**data)

    @classmethod
    def load_json(cls, filepath: str, _seen: Optional[set] = None) -> 'BasicConfig':
        _seen = set() if _seen is None else _seen
        ap = os.path.abspath(filepath)
        if ap in _seen:
            raise ValueError(f'circular _base_ inheritance detected at {filepath}')
        _seen.add(ap)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "_base_" in data:
            base_path = data.pop("_base_")
            base_dir = os.path.dirname(filepath)
            full_base_path = os.path.join(base_dir, base_path)
            base_config = cls.load_json(full_base_path, _seen)
            base_config.update(data)
            return base_config
        return cls.from_dict(data)

    def save_json(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load_yaml(cls, filepath: str, _seen: Optional[set] = None) -> 'BasicConfig':
        if yaml is None:
            raise ImportError("PyYAML is required to load yaml configs. Run 'pip install pyyaml'")
        _seen = set() if _seen is None else _seen
        ap = os.path.abspath(filepath)
        if ap in _seen:
            raise ValueError(f'circular _base_ inheritance detected at {filepath}')
        _seen.add(ap)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if "_base_" in data:
            base_path = data.pop("_base_")
            base_dir = os.path.dirname(filepath)
            full_base_path = os.path.join(base_dir, base_path)
            base_config = cls.load_yaml(full_base_path, _seen)
            base_config.update(data)
            return base_config
        return cls.from_dict(data)

    def save_yaml(self, filepath: str) -> None:
        if yaml is None:
            raise ImportError("PyYAML is required to save yaml configs. Run 'pip install pyyaml'")
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def load_from_env(self, prefix: str = "CONFIG_") -> 'BasicConfig':
        self._check_frozen()
        for env_name, env_val in os.environ.items():
            if env_name.startswith(prefix):
                config_path = env_name[len(prefix):].lower().replace('__', '.')
                typed_val = self._parse_string_value(env_val)
                self.set_by_path(config_path, typed_val)
        return self

    def update_from_args(self) -> 'BasicConfig':
        self._check_frozen()
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith('--'):
                if '=' in arg:
                    key, val_str = arg[2:].split('=', 1)
                    typed = self._parse_string_value(val_str)
                else:
                    key = arg[2:]
                    nxt = args[i + 1] if i + 1 < len(args) else None
                    if nxt is not None and not nxt.startswith('--'):
                        typed = self._parse_string_value(nxt)
                        i += 1
                    else:
                        typed = True
                self.set_by_path(key, typed)
            i += 1
        return self

    def flatten(self, parent_key: str = '', sep: str = '.', resolve: bool = True) -> Dict[str, Any]:
        items = []
        for k, v in self._data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, BasicConfig):
                items.extend(v.flatten(new_key, sep=sep, resolve=resolve).items())
            else:
                items.append((new_key, self.resolve_value(v) if resolve else v))
        return dict(items)

    def diff(self, base_cfg: 'BasicConfig') -> Dict[str, Dict[str, Any]]:
        diffs = {}
        flat_self = self.flatten()
        flat_base = base_cfg.flatten()
        for k, v in flat_self.items():
            if k not in flat_base:
                diffs[k] = {"current": v, "base": None}
            elif flat_base[k] != v:
                diffs[k] = {"current": v, "base": flat_base[k]}
        for k, v in flat_base.items():
            if k not in flat_self:
                diffs[k] = {"current": None, "base": v}
        return diffs

    def get_hash(self, resolve: bool = True) -> str:
        flat = self.flatten(resolve=resolve)
        sorted_str = json.dumps(flat, sort_keys=True, default=str)
        return hashlib.md5(sorted_str.encode('utf-8')).hexdigest()[:8]

    def to_safe_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self._data.items():
            if any(sec in key.lower() for sec in self._SECRET_KEYS):
                result[key] = "******"
            elif isinstance(value, BasicConfig):
                result[key] = value.to_safe_dict()
            else:
                result[key] = self.resolve_value(value)
        return result

    def freeze(self) -> None:
        super().__setattr__('_frozen', True)
        for val in self._data.values():
            if isinstance(val, BasicConfig):
                val.freeze()

    def defrost(self) -> None:
        super().__setattr__('_frozen', False)
        for val in self._data.values():
            if isinstance(val, BasicConfig):
                val.defrost()

    def _check_frozen(self) -> None:
        if getattr(self, '_frozen', False):
            raise TypeError("This BasicConfig instance is frozen and cannot be modified.")

    def watch(self, key: str, callback: Callable[[Any], None]) -> None:
        if not hasattr(self, '_callbacks'):
            super().__setattr__('_callbacks', {})
        self._callbacks[key] = callback

    def _trigger_callback(self, key: str, value: Any) -> None:
        if hasattr(self, '_callbacks') and key in self._callbacks:
            self._callbacks[key](self.resolve_value(value))

    def __getattr__(self, key: str) -> Any:
        if key.startswith('_') or key == '_data':
            return super().__getattribute__(key)
        if key in self._data:
            val = self._data[key]
            if isinstance(val, BasicConfig):
                return val
            return self.resolve_value(val)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith('_'):
            super().__setattr__(key, value)
            return
        self._check_frozen()
        schema = getattr(self, '__schema__', {})
        target_cls = schema.get(key)
        if isinstance(value, dict):
            if isinstance(target_cls, type) and issubclass(target_cls, BasicConfig):
                if key in self._data and isinstance(self._data[key], target_cls):
                    self._data[key].update(value)
                else:
                    self._data[key] = target_cls(**value)
            elif key in self._data and isinstance(self._data[key], BasicConfig):
                self._data[key].update(value)
            else:
                self._data[key] = BasicConfig(**value)
        else:
            self._data[key] = value
        self._trigger_callback(key, value)

    def __delattr__(self, key: str) -> None:
        self._check_frozen()
        if key in self._data:
            del self._data[key]
        else:
            raise AttributeError(f"'{key}' not found")

    def __getitem__(self, key: str) -> Any:
        return self.resolve_value(self._data[key])

    def __setitem__(self, key: str, value: Any) -> None:
        self._check_frozen()
        if isinstance(value, dict):
            if key in self._data and isinstance(self._data[key], BasicConfig):
                self._data[key].update(value)
            else:
                self._data[key] = BasicConfig(**value)
        else:
            self._data[key] = value
        self._trigger_callback(key, value)

    def __delitem__(self, key: str) -> None:
        self._check_frozen()
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        if key in self._data:
            return True
        cur = self
        for k in key.split('.'):
            if isinstance(cur, BasicConfig) and k in cur._data:
                cur = cur._data[k]
            else:
                return False
        return True

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return json.dumps(self.to_safe_dict(), indent=4, ensure_ascii=False)

    @classmethod
    def from_dataclass(cls, dc) -> 'BasicConfig':
        import dataclasses as _dc

        try:
            hints = _tp.get_type_hints(type(dc))
        except Exception:
            hints = {}

        schema, data = {}, {}
        for f in _dc.fields(dc):
            val = getattr(dc, f.name)
            if _dc.is_dataclass(val) and not isinstance(val, type):
                val = cls.from_dataclass(val)
                schema[f.name] = type(val)
            else:
                schema[f.name] = _ann_to_rule(hints.get(f.name))
            data[f.name] = val
        obj = cls(**data)
        object.__setattr__(obj, '__schema__', schema)
        return obj


# @configclass decorator
def configclass(cls: type) -> type:
    """
    把带注解的类转成类型化 BasicConfig 子类。

    - GRPOConfig.load_yaml('file.yaml') -> GRPOConfig 实例（不是 BasicConfig）
    - 自动推导 schema；load 时 validate()
    - 缺失的 yaml 键补默认值
    - 构造后调用 __post_init__()
    - 支持 @configclass 类之间的继承

    关键修复：
    (A) 装饰时会清掉原始类上会遮蔽 __getattr__ 读取 _data 的类级默认值，
        否则 cfg.batch_size 永远返回类属性里的默认值，kwargs/yaml 覆盖全失效。
    (B) 嵌套 BasicConfig 默认值经 _copy_default 转 dict，避免共享实例被
        update 就地合并污染。
    """

    annotations: Dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        if base is object:
            continue
        annotations.update(getattr(base, '__annotations__', {}))

    raw_defaults: Dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        if base is object or base is cls:
            continue
        inherited = getattr(base, '__config_defaults__', None)
        if inherited:
            raw_defaults.update(inherited)
    for name in annotations:
        if name in cls.__dict__:
            attr = cls.__dict__[name]
            if not isinstance(attr, _FieldSpec) and not callable(attr):
                raw_defaults[name] = attr

    schema: Dict[str, Any] = {}
    defaults: Dict[str, Any] = {}
    factories: Dict[str, Callable] = {}

    for name, ann in annotations.items():
        rule = _ann_to_rule(ann)
        attr = cls.__dict__.get(name)
        if isinstance(attr, _FieldSpec):
            if attr.validator is not None:
                rule = attr.validator
            if attr.default_factory is not None:
                factories[name] = attr.default_factory
            elif attr.default is not ...:
                defaults[name] = _copy_default(attr.default)
        elif name in raw_defaults:
            defaults[name] = _copy_default(raw_defaults[name])
        schema[name] = rule

    for name in list(cls.__dict__):
        if name in annotations:
            attr = cls.__dict__[name]
            if isinstance(attr, _FieldSpec) or (not callable(attr) and name in defaults):
                delattr(cls, name)

    post_init = getattr(cls, '__post_init__', None)

    def __init__(self, **kwargs):
        BasicConfig.__init__(self)
        self.update(defaults)
        for fname, factory in factories.items():
            if fname not in kwargs and fname not in defaults:
                self._data[fname] = factory()
        self.update(kwargs)
        if callable(post_init):
            post_init(self)

    @classmethod
    def _from_dict(cls_, data: Dict[str, Any]):
        instance = cls_(**data)
        instance.validate()
        return instance

    @classmethod
    def _load_yaml(cls_, filepath: str, _seen: Optional[set] = None):
        result = BasicConfig.load_yaml.__func__(cls_, filepath, _seen)
        if callable(post_init):
            post_init(result)
        result.validate()
        return result

    @classmethod
    def _load_json(cls_, filepath: str, _seen: Optional[set] = None):
        result = BasicConfig.load_json.__func__(cls_, filepath, _seen)
        if callable(post_init):
            post_init(result)
        result.validate()
        return result

    namespace = {
        '__module__': cls.__module__,
        '__qualname__': cls.__qualname__,
        '__doc__': cls.__doc__,
        '__schema__': schema,
        '__defaults__': defaults,
        '__config_defaults__': defaults,
        '__factories__': factories,
        '__config_annotations__': annotations,
        '__init__': __init__,
        '__post_init__': post_init,
        'from_dict': _from_dict,
        'load_yaml': _load_yaml,
        'load_json': _load_json,
        '__config_original__': cls,
    }

    new_cls = type(cls.__name__, (cls, BasicConfig), namespace)
    return new_cls