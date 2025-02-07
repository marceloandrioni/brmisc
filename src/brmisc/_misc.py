__all__ = [
    "kwargs2attrs",
    "random_str",
    "Outfile",
    "Timeit",
    "timeit",
]


import sys
from typing import Callable, Annotated
import random
import string
from functools import wraps
import datetime
from pathlib import Path
import pprint
import re

from pydantic import Field
from .type_validation import validate_types_in_func_call


class kwargs2attrs:

    def __init__(self, **kwargs) -> None:
        """
        A utility class to dynamically convert keyword arguments (kwargs) into instance attributes.

        Parameters
        ----------
        **kwargs : Any
            Arbitrary keyword arguments used to populate instance attributes.

        Examples
        --------
        >>> obj = kwargs2attrs(name="John", age=30)
        >>> obj.name
        'John'
        >>> obj.age
        30

        """

        # allow only names starting with a-zA-Z or underscore (_)
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        for kwarg in kwargs:
            if not re.fullmatch(pattern, kwarg):
                raise ValueError(f"string '{kwarg}' cant be used as a attribute name")

        self.__dict__.update(**kwargs)

    def __repr__(self) -> str:
        return pprint.pformat(self.__dict__)


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

    return "".join(random.choices(sample_space, k=n))


class Outfile:

    # @todo: option to check if the file type is correct using magic or a func, e.g.:
    #
    # Outfile(..., check_file="netcdf")
    #
    # def my_func(file):
    #     with xr.open_dataset(file) as ds:
    #         if len(ds["time"]) < 24:
    #             raise ValueError()
    #
    # Outfile(..., check_file=my_func)

    @validate_types_in_func_call
    def __init__(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
        use_temporary_file: bool = True,
        delete_temporary_file_on_error: bool = True,
        mandatory_extension: Annotated[str, Field(pattern=r"^\.[a-zA-Z0-9]+")]
        | None = None,
    ) -> None:
        """A context manager for safely handling file creation, optionally using
        a temporary file during the process. This class ensures that the target
        file is only finalized once the operations inside the context block are
        successful.

        Parameters
        ----------
        path : str | Path
            The path where the output file will be created.
        overwrite : bool, optional
            Whether to overwrite the target file if it already exists.
            Defaults to False.
        use_temporary_file : bool, optional
            Whether to use a temporary file before finalizing the output file.
            Defaults to True.
        delete_temporary_file_on_error : bool, optional
            If an error occurs during the context block, deletes the temporary
            file. Defaults to True.
        mandatory_extension : str or None, optional
            Enforces a specific file extension for the output file.

        Examples
        --------
        Use `Outfile` to safely create a file, leveraging a temporary file for
        atomic writes:

        >>> with Outfile("/tmp/some/dir/myfile.txt") as outfile:
        ...     with open(outfile, "w") as fp:
        ...         print(f"Writing data to temporary file: {outfile}")
        ...         fp.write("Hello!")

        After the context block completes successfully, the temporary file (if
        used) will be renamed to the final target file. If an error occurs, the
        temporary file will be deleted (if `delete_temporary_file_on_error` is True).

        """

        self.path = Path(path)
        self.overwrite = overwrite
        self.use_temporary_file = use_temporary_file
        self.delete_temporary_file_on_error = delete_temporary_file_on_error

        if mandatory_extension and mandatory_extension != self.path.suffix:
            raise ValueError(
                f"File '{self.path}' does not have the mandatory extension '{mandatory_extension}'"
            )

    def __enter__(self) -> Path:
        if self.path.exists() and not self.overwrite:
            raise ValueError(f"File '{self.path}' exists.")

        # create parent dir if it doesn't exist
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._temp = self.path
        if self.use_temporary_file:
            self._temp = self.path.with_suffix(
                f".tmp{random_str()}{self.path.suffix}"
            )

        return self._temp

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_val is not None:
            if self.delete_temporary_file_on_error:
                self._temp.unlink()
            raise exc_val

        # another process may have created self.path while the with block
        # was being executed, so check again if file exits.
        if self.path.exists() and not self.overwrite:
            raise ValueError(f"File {self.path} exists.")

        self._temp.rename(self.path)


class Timeit:

    @validate_types_in_func_call
    def __init__(
        self,
        *,
        display_msg: bool = True,
        msg_template: Annotated[str, Field(pattern=r"\{time_delta\}")] = "Execution time: {time_delta}",
    ) -> None:
        """
        A context manager for measuring the execution time of a block of code,
        with optional display of the elapsed time.

        Parameters
        ----------
        display_msg : bool, optional
            Whether to print the execution time message after the block has
            been executed. Defaults to True.
        msg_template : str, optional
            Message template. {time_delta} will be replaced by the actual
            execution time.

        Examples
        --------
        >>> with Timeit() as timer:
        ...     time.sleep(2)
        Execution time: 0:00:02.000123

        >>> with Timeit(msg_template="with block took: {time_delta}") as timer:
        ...     time.sleep(2)
        with block took: 0:00:02.012325

        """

        self.display_msg = display_msg
        self.msg_template = msg_template
        self.dt_start: datetime.datetime | None = None
        self.dt_stop: datetime.datetime | None = None
        self.time_delta: datetime.timedelta | None = None

    @staticmethod
    def _utcnow() -> datetime.datetime:
        # Note: datetime.datetime.utcnow is deprecated and will be removed
        return datetime.datetime.now(datetime.timezone.utc)

    def __enter__(self) -> None:
        self.dt_start = self._utcnow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.dt_stop = self._utcnow()
        self.time_delta = self.dt_stop - self.dt_start

        if exc_val is not None:
            print(f"Error after: {self.time_delta}", file=sys.stderr)
            raise exc_val

        print(self.msg_template.format(time_delta=self.time_delta))


def timeit(f: Callable) -> Callable:
    """Decorator to measure the execution time of a function.

    Examples
    --------
    >>> import time
    >>> @timeit
    ... def hello(msg, sleep):
    ...     print(msg)
    ...     time.sleep(sleep)
    >>> hello("Hello World", sleep=1)
    Running function: hello('Hello World', sleep=1)
    Hello World
    Execution time: 0:00:01.005325

    """

    @wraps(f)
    def wrap(*args, **kwargs):

        # string representation for args and kwargs
        # Note: repr(x) is better than str(x) because shows string between quotes
        args_lst = [repr(x) for x in args]
        kwargs_lst = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        args_str = ", ".join(args_lst + kwargs_lst)

        print(f"Running function: {f.__name__}({args_str})")
        with Timeit() as _:
            result = f(*args, **kwargs)

        return result

    return wrap
