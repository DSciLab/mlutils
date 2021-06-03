import copy
from collections import defaultdict
import inspect
from typing import Any, Callable, List, Optional, Union


__all__ = ['get', 'register', 'all']
__MODS__ = defaultdict(lambda: {})


def get(group_name: str, name: Optional[str]=None) -> Any:
    """
        :param group_name(str): group name of the class or function belong to.
        :param name(str): module name or alias name for the class or function
        >>> f = get('test_group', 'f_function')
        >>> f()
        >>> #============
        >>> A = get('test_group', 'a_function')
        >>> a = A()
    """
    if group_name not in __MODS__.keys():
        raise RuntimeError(
            f'No such mod group named ({group_name}).')
    group = __MODS__[group_name]

    if name is None:
        return group

    if name not in group.keys():
        raise RuntimeError(
            f'No such mod named ({name}) in group ({group_name}).')
    return group[name]


def add(group_name: str,
        mod: Any, *,
        alias: Optional[Union[str, List[str]]]=None
) -> None:
    mod_name = mod.__name__
    __MODS__[group_name][mod_name] = mod
    if alias is not None:
        if not isinstance(alias, (list, tuple)):
            alias = [alias]
        for alias_item in alias:
            __MODS__[group_name][alias_item] = mod


def all() -> dict:
    return dict(copy.copy(__MODS__))


def register(group_name: str, **kwargs) -> Callable:
    """
        :param group_name(str): group name of the class or function belong to.
        :param alias(str): bind name for the class or function,
            default: the class name or the function name
        >>> @register('test_group', alias='f_function')
        >>> def f(*args, **kwargs):
        >>>     pass
        >>> #=============
        >>> @register('test_group', alias='a_class')
        >>> class A:
        >>>     def __init__(self):
        >>>         pass
    """
    def _exec(func):
        if inspect.isfunction(func) or inspect.isclass(func):
            alias_name = kwargs.get('alias', None)
            mod_name = func.__name__
            __MODS__[group_name][mod_name] = func
            if alias_name is not None:
                if not isinstance(alias_name, (list, tuple)):
                    alias_name = [alias_name]

                for alias_name_item in alias_name:
                    __MODS__[group_name][alias_name_item] = func
        def __exec(*args, **kwargs):
            return func(*args, **kwargs)
        return __exec
    return _exec


if __name__ == '__main__':
    # test
    @register('test_func')
    def f(a):
        print(a)

    ff = get('test_func', 'f')
    ff(2)

    print('-' * 20)
    try:
        ff = get('undefined_group', 'f')
        print('ERROR')
    except RuntimeError:
        print('OK')

    try:
        ff = get('test_func', 'ff')
        print('ERROR')
    except RuntimeError:
        print('OK')

    print('-' * 20)
    
    @register('test_cls')
    class A(object):
        def __init__(self, a) -> None:
            super().__init__()
            print('init A', a)

    AA = get('test_cls', 'A')
    aa = AA(1)

    print('-' * 20)
    A(2)

    print('-' * 20)
    @register('test_func', alias='f2')
    def f22(a):
        print(a)

    ff = get('test_func', 'f2')
    ff(8)
