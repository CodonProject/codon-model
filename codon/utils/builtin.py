
def string_has(content: str, keywords:list[str], strict:bool=False) -> bool:
    if not strict: content = content.lower()
    for k in keywords:
        if not strict: k = k.lower()
        if k in content: return True
    return False