from collections import defaultdict
import inspect


__all__ = ['get', 'register']
__MODS__ = defaultdict(lambda: {})


def get(group_name, bind_name):
    """
        group_name(str): group name of the class or function belong to.
        bind_name(str): bind name for the class or function
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
    if bind_name not in group.keys():
        raise RuntimeError(
            f'No such mod named ({bind_name}) in group ({group_name}).')
    return group[bind_name]


def register(group_name, **kwargs):
    """
        group_name(str): group name of the class or function belong to.
        bind_name(str): bind name for the class or function,
            default: the class name or the function name
        >>> @register('test_group', bind_name='f_function')
        >>> def f(*args, **kwargs):
        >>>     pass
        >>> #=============
        >>> @register('test_group', bind_name='a_class')
        >>> class A:
        >>>     def __init__(self):
        >>>         pass
    """
    def _exec(func):
        if inspect.isfunction(func) or inspect.isclass(func):
            bind_name = kwargs.get('bind_name', None)
            if bind_name is None:
                bind_name = func.__name__
            __MODS__[group_name][bind_name] = func
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
    @register('test_func', bind_name='f2')
    def f22(a):
        print(a)

    ff = get('test_func', 'f2')
    ff(8)
