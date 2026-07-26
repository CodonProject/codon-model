from codon import *
import json
import os
import re
import hashlib
import sys

try:
    import yaml
except ImportError:
    yaml = None


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
        if val_str.lower() == 'true': return True
        if val_str.lower() == 'false': return False
        if val_str.lower() == 'none' or val_str == '': return None
        try:
            return float(val_str) if '.' in val_str else int(val_str)
        except ValueError:
            return val_str

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
            if k not in current._data:
                schema = getattr(current, '__schema__', {})
                target_cls = schema.get(k)
                if isinstance(target_cls, type) and issubclass(target_cls, BasicConfig):
                    current._data[k] = target_cls()
                else:
                    current._data[k] = BasicConfig()
            current = current._data[k]
        final_key = keys[-1]
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
            val = self.get(path) if '.' in path else self._data.get(path)
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
    def load_json(cls, filepath: str) -> 'BasicConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "_base_" in data:
            base_path = data.pop("_base_")
            base_dir = os.path.dirname(filepath)
            full_base_path = os.path.join(base_dir, base_path)
            base_config = cls.load_json(full_base_path)
            base_config.update(data)
            return base_config
        return cls.from_dict(data)

    def save_json(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load_yaml(cls, filepath: str) -> 'BasicConfig':
        if yaml is None:
            raise ImportError("PyYAML is required to load yaml configs. Run 'pip install pyyaml'")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if "_base_" in data:
            base_path = data.pop("_base_")
            base_dir = os.path.dirname(filepath)
            full_base_path = os.path.join(base_dir, base_path)
            base_config = cls.load_yaml(full_base_path)
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
                else:
                    key = arg[2:]
                    if i + 1 < len(args) and not args[i+1].startswith('--'):
                        val_str = args[i+1]
                        i += 1
                    else:
                        raise ValueError(f"CLI Argument value missing for key: '{arg}'")
                typed_val = self._parse_string_value(val_str)
                self.set_by_path(key, typed_val)
            i += 1
        return self

    def flatten(self, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        items = []
        for k, v in self._data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, BasicConfig):
                items.extend(v.flatten(new_key, sep=sep).items())
            else:
                items.append((new_key, self.resolve_value(v)))
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

    def get_hash(self) -> str:
        flat_dict = self.flatten()
        sorted_str = json.dumps(flat_dict, sort_keys=True)
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
        if key in ('_data', '_frozen', '_callbacks'):
            super().__setattr__(key, value)
        else:
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
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return json.dumps(self.to_safe_dict(), indent=4, ensure_ascii=False)
