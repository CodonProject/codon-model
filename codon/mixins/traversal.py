class TraversalMixin:
    def trigger(self, func_name: str, *args, **kwargs) -> None:
        for module in self.modules():
            if hasattr(module, func_name): 
                func = getattr(module, func_name)
                if callable(func): func(*args, **kwargs)