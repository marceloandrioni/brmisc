from __future__ import annotations


__all__ = [
    "kwargs2attrs",
    "random_str",
    "Outfile",
    "Timeit",
    "timeit",
    "evaluate_operation",
    "raise_if_operation_is_false",
    "ListOfObjs",
]


import sys
from typing import Callable, Annotated, Any, Literal, Sequence, TypeVar
import random
import string
from functools import wraps
import datetime
from pathlib import Path
import pprint
import re
import operator

from pydantic import Field
from .type_validation import validate_types_in_func_call


class kwargs2attrs:
    """A utility class to dynamically convert keyword arguments (kwargs) into instance attributes.

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

    def __init__(self, **kwargs) -> None:

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
    """A context manager for safely handling file creation, optionally using a
    temporary file during the process. This class ensures that the target file
    is only finalized once the operations inside the context block are successful.

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

    After the context block completes successfully, the temporary file (if used)
    will be renamed to the final target file. If an error occurs, the temporary
    file will be deleted (if `delete_temporary_file_on_error` is True).

    """

    # @todo: option to check if the file type is correct using file magic or a func, e.g.:
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
        mandatory_extension: Annotated[str, Field(pattern=r"^\.[a-zA-Z0-9]+")] | None = None,
    ) -> None:

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
    """A context manager for measuring the execution time of a block of code,
    with optional display of the elapsed time.

    Parameters
    ----------
    display_msg : bool, optional
        Whether to print the execution time message after the block has been
        executed. Defaults to True.
    msg_template : str, optional
        Message template. {time_delta} will be replaced by the actual execution
        time.

    Examples
    --------
    >>> with Timeit() as timer:
    ...     time.sleep(2)
    Execution time: 0:00:02.000123

    >>> with Timeit(msg_template="with block took: {time_delta}") as timer:
    ...     time.sleep(2)
    with block took: 0:00:02.012325

    """

    @validate_types_in_func_call
    def __init__(
        self,
        *,
        display_msg: bool = True,
        msg_template: Annotated[str, Field(pattern=r"\{time_delta\}")] = "Execution time: {time_delta}",
    ) -> None:

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


@validate_types_in_func_call
def evaluate_operation(
    left_side_value: Any,
    operation: Literal["==", "!=", ">", ">=", "<", "<=", "is", "is not"],
    right_side_value: Any,
    /,
) -> bool:
    """Evaluate a comparison operation between two values.

    This function takes two values and a specified comparison operation,
    and returns the result of the operation as a boolean.

    Parameters:
    ----------
    left_side_value : Any
        The value on the left side of the comparison.
    operation : str
        A string representing the comparison operation to be performed.
        Supported operations are:
            - "==": equal to
            - "!=": not equal to
            - ">": greater than
            - ">=": greater than or equal to
            - "<": less than
            - "<=": less than or equal to
            - "is"
            - "is not"
    right_side_value : Any
        The value on the right side of the comparison.

    Returns:
    -------
    bool
        The result of the comparison operation. Returns True if the
        comparison is valid, otherwise returns False.

    Example:
    --------
    >>> evaluate_operation(5, ">", 3)
    True

    >>> evaluate_operation(4, "==", 5)
    False

    """

    operations = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "is": operator.is_,
        "is not": operator.is_not,
    }

    return operations[operation](left_side_value, right_side_value)


@validate_types_in_func_call
def raise_if_operation_is_false(
    left_side_name: str,
    left_side_value: Any,
    operation: Literal["==", "!=", ">", ">=", "<", "<=", "is", "is not"],
    right_side_name: str,
    right_side_value: Any,
    /,
) -> None:
    """Raise a ValueError if the specified comparison operation evaluates to False.

    This function takes two values along with their names and a specified
    comparison operation. If the operation evaluates to False, it raises
    a ValueError with a message indicating the nature of the failure.

    Parameters:
    ----------
    left_side_name : str
        The name of the left side value, used for error reporting.
    left_side_value : Any
        The value on the left side of the comparison.
    operation : str
        A string representing the comparison operation to be performed.
        Supported operations are:
            - "==": equal to
            - "!=": not equal to
            - ">": greater than
            - ">=": greater than or equal to
            - "<": less than
            - "<=": less than or equal to
            - "is"
            - "is not"
    right_side_name : str
        The name of the right side value, used for error reporting.
    right_side_value : Any
        The value on the right side of the comparison.

    Returns:
    -------
    None
        The function does not return any value. It either completes
        successfully or raises a ValueError.

    Raises:
    -------
    ValueError
        If the comparison operation evaluates to False, a ValueError
        is raised with a message indicating the names of the values
        involved in the operation.

    Example:
    --------
    >>> raise_if_operation_is_false("a", 5, ">", "b", 3)  # No exception
    >>> raise_if_operation_is_false("a", 5, "<", "b", 3)
    ValueError: Operation 'a' < 'b' is False

    """

    if evaluate_operation(left_side_value, operation, right_side_value):
        return

    # Note: left/rigth_side_values can be anything, so don't shown the values
    # in the error message, only the names
    err_msg = f"Operation '{left_side_name}' {operation} '{right_side_name}' is False"
    raise ValueError(err_msg)


T = TypeVar("T")


class ListOfObjs(list):
    """A custom list implementation that holds objects of a specified class type
    and ensures that each object has a specific attribute (id_field).

    Parameters:
    ----------
    iterable : Sequence (int, tuple, etc) or None
    id_field : str
        The name of the attribute that should be present in each object.
    class_def : class or None
        An optional class definition that specifies the type
        of objects to be stored in the list. If not provided, the type will
        be inferred from the first item in the iterable.
    unique : bool
        A flag indicating whether the values of id_field must be unique in the list.
        Defaults to True.

    Example:
    --------

    >>> @dataclass
    ... class Student:
    ...     name: str
    ...     age: int

    >>> @dataclass
    ... class Teacher:
    ...     name: str
    ...     age: int

    >>> student1 = Student("John", 10)
    >>> student2 = Student("Bob", 11)
    >>> student3 = Student("John", 12)
    >>> teacher1 = Teacher("Mary", 30)

    Initialize ListOfObjs with an Iterable:

    >>> students = ListOfObjs((student1, student2), id_field="name")

    >>> students = ListOfObjs((student1, student2, teacher1), id_field="name")
    TypeError: Object must be of type 'Student'

    Initialize a empty ListOfObjs and then append

    >>> students = ListOfObjs(id_field="name", class_def=Student)
    >>> students.append(student1)
    >>> students.append(student2)

    >>> students = ListOfObjs((student1, student2, student3), id_field="name")
    ValueError: Value 'John' appears more than once.

    >>> students = ListOfObjs((student1, student2, student3), id_field="name", unique=False)

    """

    # Note: use iterable: Sequence[T] instead of iterable: Iterable[T] due to
    # https://github.com/pydantic/pydantic/issues/9541
    @validate_types_in_func_call
    def __init__(
        self,
        iterable: Sequence[T] | None = None,
        *,
        id_field: Annotated[str, Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")],
        class_def: T | None = None,
        unique: bool = True,
    ) -> None:

        self._id_field = id_field

        if not any((iterable, class_def)):
            raise ValueError("iterable OR class_def must be given")

        # get type from class_def or from the first item in iterable
        #
        # Note: hasattr(class_def, id_field) returns False for classes
        # defined with dataclass, pydantic.BaseModel, etc. Therefore, the
        # presence of attribute id_field should be checked in the intances
        self._class_def = type(next(iter(iterable))) if class_def is None else class_def

        self._unique = unique

        super().__init__([])
        for item in iterable or []:
            self.append(item)

    def __repr__(self) -> str:
        return self.ids.__repr__()

    def _validate_item_type(self, item: T) -> None:

        if not isinstance(item, self._class_def):
            raise TypeError(f"Object must be of type '{self._class_def.__name__}'")

        if not hasattr(item, self._id_field):
            raise TypeError(f"Object must have attribute '{self._id_field}'")

    def _validate_iterable_type(self, value: Sequence[T]) -> None:

        # Note:
        # https://docs.pydantic.dev/2.10/errors/usage_errors/#invalid-self-type
        # at the current version (2.10.6) pydantic does not allow Self validation
        #
        # @validate_types_in_func_call
        # def func(self, value: Self) -> Self:
        #   ....
        #
        # So checking to make sure value/iterable is of type ListOfObjs.
        if not isinstance(value, type(self)):
            err_msg = (
                f"can only concatenate {self.__class__.__name__}"
                f" (not '{value.__class__.__name__}') to {self.__class__.__name__}"
            )
            raise TypeError(err_msg)

    @staticmethod
    def _raise_if_non_unique(itens: list[Any], new_item: Any) -> None:
        """Raise if "itens + [new_item]" has non unique values."""

        if set(itens + [new_item]) == set(itens):
            raise ValueError(f"Value '{new_item}' appears more than once.")

    def __setitem__(self, index: int, item: T) -> None:
        self._validate_item_type(item)

        if self._unique:
            ids = self.ids
            ids.pop(index)
            self._raise_if_non_unique(ids, getattr(item, self._id_field))

        super().__setitem__(index, item)

    def __add__(self, iterable: "ListOfObjs") -> "ListOfObjs":
        self._validate_iterable_type(iterable)
        x = self.copy()
        for item in iterable:
            x.append(item)
        return x

    def append(self, item: T, /) -> None:
        idx = len(self)
        self.insert(idx, item)

    def count(self, value: Any, /) -> int:
        return self.ids.count(value)

    def extend(self, iterable: Sequence[T], /) -> "ListOfObjs":
        for item in iterable:
            self.append(item)

    def index(self, value: Any, /) -> int:
        return self.ids.index(value)

    def insert(self, index: int, item: T, /) -> None:
        self._validate_item_type(item)

        if self._unique:
            ids = self.ids
            self._raise_if_non_unique(ids, getattr(item, self._id_field))

        super().insert(index, item)

    def remove(self, value: Any, /) -> None:
        idx = self.ids.index(value)
        self.pop(idx)

    def sort(self, *, reverse: bool = False) -> None:
        super().sort(
            key=lambda item: getattr(item, self._id_field),
            reverse=reverse,
        )

    @property
    def ids(self) -> list[Any]:
        return [getattr(item, self._id_field) for item in self]

    def get(self, value: Any, /) -> T:
        return self[self.ids.index(value)]
