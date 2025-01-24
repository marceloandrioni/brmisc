__all__ = [
    "validate_types_in_func_call",
    "validate_type",
    "int_like",
    "datetime_like",
    "datetime_like_naive",
    "datetime_like_aware",
    "datetime_like_utc",
    "path_like",
]


from typing import Annotated, Any
from pydantic import (
    validate_call, BeforeValidator, AfterValidator, TypeAdapter, ConfigDict,
    Field, NaiveDatetime, AwareDatetime)
import numbers
import datetime
import pytz
from itertools import accumulate
import numpy as np
import pandas as pd
from pathlib import Path


# Note:
# Pydantic is too lax with parsing, so always use strict=True.
# Arbitrary_types_allowed is needed for numpy, pandas, xarray, etc.
CONFIG = ConfigDict(
    strict=True,
    arbitrary_types_allowed=True,
)

# Note: it is possible to set strict=True behavior in a arg-by-arg approach, e.g.:
#
# >>> @validate_call
# ... def func(
# ...     x: Annotated[int, Field(strict=True)],
# ...     y: Annotated[float, Field(strict=True)],
# ... ) -> None:
# ...     print(x * y)
#
# but as in most cases strict=True is the desired behavior, this will add a
# lot of boilerplate when compared to the decorator approach
#
# >>> @validate_call(config=dict(strict=True))
# ... def func(x: int, y: float) -> None:
# ...     print(x * y)
#
# so leave strict mode on to turn off when needed, e.g.:
#
# >>> @validate_call(config=dict(strict=True))
# ... def func(
# ...     x: int,
# ...     y: Annotated[float, Field(strict=False)],
# ...     ) -> None:
# ...     print(x * y)
# >>> func(1, "2")
# 2.0

validate_types_in_func_call = validate_call(
    config={
        **CONFIG,
        "validate_default": True,
    },
    validate_return=True,
)


validate_types_in_func_call.__doc__ = (
    """Returns a decorator to enforce type checking on the arguments and return
    value of a function.

    Examples
    --------
    >>> @validate_types_in_func_call
    ... def func(x: int, y: str) -> tuple[int, str]:
    ...     return x, y

    >>> func(1, "abc")
    (1, 'abc')

    >>> func(1, 2)
    ValidationError: 1 validation error for func
    1
      Input should be a valid string [type=string_type, input_value=2,
    input_type=int]

    """
)


def validate_type(value: Any, type: Any = Any) -> Any:
    """Validate type.

    Parameters
    ----------
    value : Any
    type : Any
        Type of value.

    Returns
    -------
    value : Any
        Value with type `type`.

    Examples
    --------
    >>> validate_type(1, int)
    1

    >>> validate_type(1.0, int)
    ValidationError: 1 validation error for int
      Input should be a valid integer [type=int_type, input_value=1.0,
    input_type=float]

    >>> validate_type(1.0, Union[int, float])
    1.0

    """

    return TypeAdapter(type, config=CONFIG).validate_python(value)


# -----------------------------------------------------------------------------
def parse_round_number(value: Any) -> Any:
    if isinstance(value, numbers.Number):
        assert not (value - int(value)), f"value {value} is not a round number"
        return int(value)
    return value


int_like = Annotated[
    int,
    BeforeValidator(parse_round_number),
]
int_like.__doc__ = (
    """Type alias to validate int_like (int, round float, round decimal, etc)
    and coerce it to int.

    Examples
    --------
    >>> @validate_types_in_func_call
    ... def func(x: int_like):
    ...     return x

    >>> func(10)
    10

    >>> func(10.0)
    10

    >>> func(10.1)
    ValidationError: 1 validation error for func
    0
      Assertion failed, value 10.1 is not a round number [type=assertion_error,
    input_value=10.1, input_type=float]

    Extra validators can be added:

    >>> from pydantic import Field
    >>> int_gt_10 = Annotated[int_like, Field(gt=10)]
    >>> @validate_types_in_func_call
    ... def func(x: int_gt_10):
    ...     return x

    >>> func(9)
    ValidationError: 1 validation error for func
    0
      Input should be greater than 10 [type=greater_than, input_value=9,
    input_type=int]

    validate_type can be used to validate the value without a decorator

    >>> validate_type(10.0, int_like)
    10

    >>> validate_type(10.1, int_like)
    ValidationError: 1 validation error for function-before[parse_round_number(),
    int]
      Assertion failed, value 10.1 is not a round number [type=assertion_error,
    input_value=10.1, input_type=float]

    """
)


# -----------------------------------------------------------------------------
def parse_date(value: Any) -> Any:
    # datetime is a subclass of date, so check with type
    if type(value) is datetime.date:
        return datetime.datetime.combine(value, datetime.time())
    return value


def parse_dt64(value: Any) -> Any:
    if isinstance(value, np.datetime64):
        return pd.to_datetime(value).to_pydatetime()
    return value


def parse_timestamp(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return pd.to_datetime(value).to_pydatetime()
    return value


def datetime_formats() -> list[str]:
    """Return a list of valid datetime formats."""

    def add_fz(lst: list[str]) -> list[str]:
        return lst + [f"{lst[-1]}{x}" for x in [".%f", "%z", ".%f%z"]]

    Y_m_d_H_M_S = add_fz(list(accumulate(["%Y", "-%m", "-%d", " %H", ":%M", ":%S"])))
    Y_m_dTH_M_S = add_fz(list(accumulate(["%Y", "-%m", "-%d", "T%H", ":%M", ":%S"])))
    YmdHMS = add_fz(list(accumulate(["%Y", "%m", "%d", "%H", "%M", "%S"])))
    d_m_Y_H_M_S = add_fz(list(accumulate(["%d/%m/%Y", " %H", ":%M", ":%S"])))
    fmts = Y_m_d_H_M_S + Y_m_dTH_M_S + YmdHMS + d_m_Y_H_M_S

    # remove duplicated keeping order
    fmts = list(dict.fromkeys(fmts))

    return fmts


def parse_str(value: Any) -> Any:
    """Validate time string."""

    # Note: pydantic parses strings without "-" as seconds/miliseconds, e.g.:
    # >>> TypeAdapter(datetime.datetime).validate_python("20010203")
    # datetime.datetime(1970, 8, 20, 14, 23, 23, tzinfo=TzInfo(UTC))
    # so using a custom list of valid formats

    if isinstance(value, str):

        fmts = datetime_formats()

        for fmt in fmts:
            try:
                return datetime.datetime.strptime(value, fmt)
            except Exception as err:
                pass

        fmts_str = "\n".join(fmts)
        raise ValueError(f"time data '{value}' does not match formats:\n{fmts_str}")

    return value


def dt_naive_to_dt_utc(dt: datetime.datetime) -> datetime.datetime:
    if isinstance(dt, datetime.datetime) and dt.tzinfo is None:
        return dt.astimezone(datetime.timezone.utc)
    return dt


def dt_must_be_utc(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime timezone is not UTC."""
    if dt.tzinfo.utcoffset(dt) != datetime.timedelta(0):
        raise ValueError("datetime object must have UTC timezone")
    return dt


datetime_like = Annotated[
    datetime.datetime,
    BeforeValidator(parse_date),
    BeforeValidator(parse_dt64),
    BeforeValidator(parse_timestamp),
    BeforeValidator(parse_str),
]
datetime_like.__doc__ = (
    """Type alias to validate datetime_like (datetime, date, np.datetime64,
    pd.Timestamp, etc) and coerce it to datetime.

    Examples
    --------
    >>> @validate_types_in_func_call
    ... def func(x: datetime_like):
    ...     return x

    >>> func(datetime.datetime(2001,2,3,4,5,6))
    datetime.datetime(2001, 2, 3, 4, 5, 6)

    >>> func(datetime.date(2001,2,3))
    datetime.datetime(2001, 2, 3, 0, 0)

    >>> func(np.datetime64("2001-02-03 04:05:06"))
    datetime.datetime(2001, 2, 3, 4, 5, 6)

    >>> func(pd.Timestamp("2001-02-03 04:05:06"))
    datetime.datetime(2001, 2, 3, 4, 5, 6)

    >>> func("20010203040506")
    datetime.datetime(2001, 2, 3, 4, 5, 6)

    >>> validate_type("20010203040506", datetime_like)
    datetime.datetime(2001, 2, 3, 4, 5, 6)

    """
)


datetime_like_naive = Annotated[(
    NaiveDatetime,
    *(datetime_like.__metadata__),
)]
datetime_like_naive.__doc__ = (
    """Same as datetime_like, but only naive (without time-zone) datetimes
    are allowed.

    Examples
    --------
    >>> validate_type("2001-02-03 04:05:06", datetime_like_naive)
    datetime.datetime(2001, 2, 3, 4, 5, 6)

    >>> validate_type("2001-02-03 04:05:06Z", datetime_like_naive)
    ValidationError: 1 validation error for function-before[parse_str(),
    function-before[parse_timestamp(), function-before[parse_dt64(),
    function-before[parse_date(), datetime]]]]
      Input should not have timezone info [type=timezone_naive,
    input_value=datetime.datetime(2001, 2...o=datetime.timezone.utc),
    input_type=datetime]

    """
)


datetime_like_aware = Annotated[(
    AwareDatetime,
    *(datetime_like.__metadata__),
)]
datetime_like_aware.__doc__ = (
    """Same as datetime_like, but only aware (with time-zone) datetimes
    are allowed.

    Examples
    --------
    >>> validate_type("2001-02-03 04:05:06Z", datetime_like_aware)
    datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)

    >>> validate_type("2001-02-03 04:05:06", datetime_like_aware)
    ValidationError: 1 validation error for function-before[parse_str(),
    function-before[parse_timestamp(), function-before[parse_dt64(),
    function-before[parse_date(), datetime]]]]
      Input should have timezone info [type=timezone_aware,
    input_value=datetime.datetime(2001, 2, 3, 4, 5, 6), input_type=datetime]

    """
)


# Note: BeforeValidators are executed right-to-left, while AfterValidators are
# run left-to-rigth.
# https://docs.pydantic.dev/latest/concepts/validators/#ordering-of-validators
datetime_like_utc = Annotated[(
    AwareDatetime,
    BeforeValidator(dt_naive_to_dt_utc),   # run after all others BeforeValidators
    *(datetime_like.__metadata__),
    AfterValidator(dt_must_be_utc),
)]
datetime_like_utc.__doc__ = (
    """Same as datetime_like, but only UTC datetimes are allowed. Naive
    (without time-zone) datetimes are assumed to represent UTC time-zone.

    Examples
    --------
    >>> validate_type("2001-02-03 04:05:06", datetime_like_utc)
    datetime.datetime(2001, 2, 3, 6, 5, 6, tzinfo=datetime.timezone.utc)

    >>> validate_type("2001-02-03 04:05:06Z", datetime_like_utc)
    datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)

    >>> validate_type("2001-02-03 04:05:06+00:00", datetime_like_utc)
    datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)

    >>> validate_type("2001-02-03 04:05:06-03:00", datetime_like_utc)
    ValidationError: 1 validation error for function-after[dt_must_be_utc(),
    function-before[parse_str(), function-before[parse_timestamp(),
    function-before[parse_dt64(), function-before[parse_date(),
    function-before[dt_naive_to_dt_utc(), datetime]]]]]]
      Value error, datetime object must have UTC timezone [type=value_error,
    input_value='2001-02-03 04:05:06-03:00', input_type=str]

    """
)


# -----------------------------------------------------------------------------
# set strict=False to allow coercion from string
path_like = Annotated[Path, Field(strict=False)]
path_like.__doc__ = (
    """Type alias to validate Path, str, etc. If valid, the value is coerced to
    Path.

    Examples
    --------
    >>> validate_type(Path("/tmp/foo/bar"), path_like)
    PosixPath('/tmp/foo/bar')

    >>> validate_type("/tmp/foo/bar", path_like)
    PosixPath('/tmp/foo/bar')

    """
)

# TODO: create some more path types, e.g.:
# pydantic FilePath : must exist and be a file
# pydantic DirectoryPath: must exist and be a directory
# pydantic NewPath : must be new and parent exist (maybe turn off parent necessity)
# Path_must_be_overwritable
# check https://github.com/xfrenette/pathtype for ideas
