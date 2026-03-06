import random
import string


def safecode(length: int = 4, exclude_confusing: bool = False) -> str:
    '''
    Generates a random safe code consisting of letters and digits.

    Args:
        length (int): The length of the code to generate. Defaults to 4.
        exclude_confusing (bool): If True, excludes confusing characters
            ('0oO1iIlLq9g') to reduce human error. Defaults to False.

    Returns:
        str: The generated random code.
    '''
    characters = string.ascii_letters + string.digits

    if exclude_confusing:
        confusing_chars = '0oO1iIlLq9g'
        characters = ''.join(c for c in characters if c not in confusing_chars)

    code = ''.join(random.choices(characters, k=length))
    return code
