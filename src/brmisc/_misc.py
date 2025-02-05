__all__ = [
    "random_str",
    "timeit",
]

from typing import Callable
import random
import string
from functools import wraps
from time import time

from .type_validation import validate_types_in_func_call


@validate_types_in_func_call
def random_str(
    n: int = 8,
    lowercase: bool = True,
    uppercase: bool = True,
    digits: bool = True,
) -> str:
    """Random string.

    Parameters
    ----------
    n: int, optional
        Number of characters. A string with 8 characters and sample space of 62
        [a-zA-Z0-9] has ~218 quadrillion (62**8) possibilities.
    lowercase: bool, optional
        Use lowercase letters [a-z].
    uppercase: bool, optional
        Use uppercase letters [A-Z].
    digits: bool, optional
        Use digits [0-9].

    Returns
    -------
    out : str
        Random string with `n` charactes.

    """

    if not any((lowercase, uppercase, digits)):
        raise ValueError("At least one option must be True.")

    sample_space: list = []
    if lowercase:
        sample_space += string.ascii_lowercase
    if uppercase:
        sample_space += string.ascii_uppercase
    if digits:
        sample_space += string.digits

    return ''.join(random.choices(sample_space, k=n))


def timeit(f: Callable) -> Callable:
    """Decorator to get the execution time of a function.

    Examples
    --------
    >>> import time
    >>> @timeit
    ... def hello(msg, sleep):
    ...     print(msg)
    ...     time.sleep(sleep)
    >>> hello("Hello World", sleep=1)
    Hello World.
    hello('Hello World', sleep=1) : running time of 1.01 seconds

    """

    @wraps(f)
    def wrap(*args, **kwargs):

        t_start = time.time()
        result = f(*args, **kwargs)
        t_stop = time.time()

        # string representation for args and kwargs
        # Note: repr(x) is better than str(x) because represents string between quotes
        args_lst = [repr(x) for x in args]
        kwargs_lst = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        args_str = ", ".join(args_lst + kwargs_lst)

        msg = (f"{f.__name__}({args_str}) : running time of"
               f" {t_stop - t_start:.2f} seconds")
        print(msg)

        return result
    return wrap
