def make_layer_pattern(num_layers: int, ratio: int = 4) -> list:
    assert num_layers % ratio == 0, f'num_layers={num_layers} 必须是 {ratio} 的倍数'
    return (['gdn'] * (ratio - 1) + ['gqa']) * (num_layers // ratio)