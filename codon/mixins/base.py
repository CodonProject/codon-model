from typing import TypeVar


TOriginal = TypeVar('TOriginal')

def mixin(instance: TOriginal, *mixin_classes) -> TOriginal:
    if not mixin_classes:
        return instance

    for m in mixin_classes:
        if not isinstance(m, type):
            raise TypeError(f'Expected class type for mixin, got {type(m).__name__} instead.')

    original_class = type(instance)

    existing_bases = set(type(instance).__mro__)
    new_mixins = [m for m in mixin_classes if m not in existing_bases]
    if not new_mixins: return instance
    
    mixin_names = '_'.join(m.__name__ for m in mixin_classes)
    new_class_name = f'{original_class.__name__}_with_{mixin_names}'

    bases = (original_class,) + mixin_classes

    try:
        new_class = type(new_class_name, bases, {})
        instance.__class__ = new_class
    except TypeError as e:
        raise TypeError(f'Failed to apply mixins to instance: {e}') from e

    return instance


class CodonMixin:
    '''
    Base class for all Codon mixins.
    '''