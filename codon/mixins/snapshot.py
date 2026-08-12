class SnapshotMixin:
    def __init__(self):
        super().__init__()
        self._snapshots = {}

    @property
    def _orig(self):
        return getattr(self, 'original_model', self)

    def snapshot(self, name: str = 'default', device: str = 'cpu', lora_only: bool = False) -> None:
        from codon.utils.lora import get_lora_state_dict
        if lora_only:
            s_dict = get_lora_state_dict(self._orig)
        else:
            s_dict = self._orig.state_dict()

        self._snapshots[name] = {k: v.detach().clone().to(device=device) for k, v in s_dict.items()}

    def restore_snapshot(self, name: str = 'default', strict: bool = False) -> None:
        if name not in self._snapshots: raise KeyError(f"No snapshot found with name '{name}'")
        
        target_device = getattr(self, 'device', 'cpu')
        loaded_dict = {k: v.to(device=target_device) for k, v in self._snapshots[name].items()}
        self._orig.load_state_dict(loaded_dict, strict=strict)

    def clear_snapshots(self) -> None:
        if hasattr(self, '_snapshots'): self._snapshots.clear()