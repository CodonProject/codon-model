class SnapshotMixin:
    def __init__(self):
        super().__init__()
        self._snapshots = {}

    def snapshot(self, name: str = 'default', device: str = 'cpu') -> None:
        self._snapshots[name] = {k: v.detach().clone().to(device=device) for k, v in self.state_dict().items()}

    def restore_snapshot(self, name: str = 'default', strict: bool = True) -> None:
        if name not in self._snapshots: raise KeyError(f"No snapshot found with name '{name}'")
        
        target_device = getattr(self, 'device', 'cpu')
        self.load_state_dict({k: v.to(device=target_device) for k, v in self._snapshots[name].items()}, strict=strict)

    def clear_snapshots(self) -> None:
        if hasattr(self, '_snapshots'): self._snapshots.clear()