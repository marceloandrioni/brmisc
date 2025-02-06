__all__ = [
    "validate_types_in_func_call",
    "validate_type",
    "int_like",
    "datetime_like",
    "datetime_like_naive",
    "datetime_like_aware",
    "datetime_like_naive_or_utc_to_naive",
    "datetime_like_naive_or_utc_to_utc",
    "dt_must_be_YYYYmmdd_HHMM00",   # rounded to second
    "dt_must_be_YYYYmmdd_HH0000",   # rounded to minute
    "dt_must_be_YYYYmmdd_000000",   # rounded to midnight
    "dt_must_be_YYYYmm01_000000",   # rounded to 1st day of month
    "dt_must_be_YYYY0101_000000",   # rounded to 1st day of year
    "path_like",
    "path_like_absolute",
]


from typing import Annotated, Any
import numbers
import datetime
from itertools import accumulate
from pathlib import Path
from pydantic import (
    validate_call, BeforeValidator, AfterValidator, TypeAdapter, ConfigDict,
    Field, NaiveDatetime, AwareDatetime)
import numpy as np
import pandas as pd


# Note:
# * Pydantic is too lax with parsing, so always use strict=True.
# * Arbitrary_types_allowed is needed for numpy, pandas, xarray, etc.
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
      Assertion failed, value 10.1 is not a round number [type=assertion_error,
    input_value=10.1, input_type=float]

    Extra validators can be added:

    >>> from pydantic import Field
    >>> int_gt_10 = Annotated[int_like, Field(gt=10)]
    >>> @validate_types_in_func_call
    ... def func(x: int_gt_10):
    ...     return x

    >>> func(9)
      Input should be greater than 10 [type=greater_than, input_value=9,
    input_type=int]

    validate_type can be used to validate the value without a decorator

    >>> validate_type(10.0, int_like)
    10

    >>> validate_type(10.1, int_like)
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


def parse_timestr(value: Any) -> Any:
    """Parse time string."""

    # Note: pydantic parses strings without "-" as seconds/milliseconds, e.g.:
    # >>> TypeAdapter(datetime.datetime).validate_python("20010203")
    # datetime.datetime(1970, 8, 20, 14, 23, 23, tzinfo=TzInfo(UTC))
    # so using a custom list of valid formats

    if isinstance(value, str):

        fmts = datetime_formats()

        for fmt in fmts:
            try:
                return datetime.datetime.strptime(value, fmt)
            except ValueError as err:
                pass

        fmts_str = "\n".join(fmts)
        raise ValueError(f"time data '{value}' does not match formats:\n{fmts_str}")

    return value


def dt_naive_to_dt_utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def dt_must_be_utc(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime timezone is not UTC."""
    if dt.tzinfo.utcoffset(dt) != datetime.timedelta(0):
        raise ValueError("Input should have UTC timezone")
    return dt


def dt_utc_to_dt_naive(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is not None:
        dt_must_be_utc(dt)
        return dt.replace(tzinfo=None)
    return dt


datetime_like = Annotated[
    datetime.datetime,
    BeforeValidator(parse_date),
    BeforeValidator(parse_dt64),
    BeforeValidator(parse_timestamp),
    BeforeValidator(parse_timestr),
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
    """Same as datetime_like, but only naive datetimes are allowed.

    Examples
    --------
    >>> validate_type("2001-02-03 04:05:06", datetime_like_naive)
    datetime.datetime(2001, 2, 3, 4, 5, 6)

    >>> validate_type("2001-02-03 04:05:06Z", datetime_like_naive)
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
    """Same as datetime_like, but only aware datetimes are allowed.

    Examples
    --------
    >>> validate_type("2001-02-03 04:05:06Z", datetime_like_aware)
    datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)

    >>> validate_type("2001-02-03 04:05:06", datetime_like_aware)
      Input should have timezone info [type=timezone_aware,
    input_value=datetime.datetime(2001, 2, 3, 4, 5, 6), input_type=datetime]

    """
)


# Note: BeforeValidators are executed right-to-left, while AfterValidators are
# run left-to-rigth.
# https://docs.pydantic.dev/latest/concepts/validators/#ordering-of-validators
datetime_like_naive_or_utc_to_utc = Annotated[(
    AwareDatetime,
    BeforeValidator(dt_naive_to_dt_utc),   # run after all others BeforeValidators
    *(datetime_like.__metadata__),
    AfterValidator(dt_must_be_utc),
)]
datetime_like_naive_or_utc_to_utc.__doc__ = (
    """Same as datetime_like, but only naive and UTC datetimes are allowed.

    Naive datetimes are coerced to UTC.

    Examples
    --------
    >>> validate_type("2001-02-03 04:05:06", datetime_like_naive_or_utc_to_utc)
    datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)

    >>> validate_type("2001-02-03 04:05:06Z", datetime_like_naive_or_utc_to_utc)
    datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)

    >>> validate_type("2001-02-03 04:05:06+00:00", datetime_like_naive_or_utc_to_utc)
    datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)

    >>> validate_type("2001-02-03 04:05:06-03:00", datetime_like_naive_or_utc_to_utc)
      Value error, Input should have UTC timezone [type=value_error,
    input_value='2001-02-03 04:05:06-03:00', input_type=str]

    """
)


datetime_like_naive_or_utc_to_naive = Annotated[(
    NaiveDatetime,
    BeforeValidator(dt_utc_to_dt_naive),   # run after all others BeforeValidators
    *(datetime_like.__metadata__),
)]
datetime_like_naive_or_utc_to_naive.__doc__ = (
    """Same as datetime_like, but only naive and UTC datetimes are allowed.

    UTC datetimes are coerced to naive.

    Examples
    --------
    >>> validate_type("2001-02-03 04:05:06", datetime_like_naive_or_utc_to_naive)
    datetime.datetime(2001, 2, 3, 4, 5, 6)

    >>> validate_type("2001-02-03 04:05:06Z", datetime_like_naive_or_utc_to_naive)
    datetime.datetime(2001, 2, 3, 4, 5, 6)

    >>> validate_type("2001-02-03 04:05:06+00:00", datetime_like_naive_or_utc_to_naive)
    datetime.datetime(2001, 2, 3, 4, 5, 6)

    >>> validate_type("2001-02-03 04:05:06-03:00", datetime_like_naive_or_utc_to_naive)
      Value error, Input should have UTC timezone [type=value_error,
    input_value=datetime.datetime(2001, 2...ays=-1, seconds=75600))),
    input_type=datetime]

    """
)


DT_REPLACE_DICT: dict[str, int] = {
    "microsecond": 0,
    "second": 0,
    "minute": 0,
    "hour": 0,
    "day": 1,
    "month": 1,
}


def get_dict_until(d: dict[Any, Any], key: Any) -> dict[Any, Any]:

    # raise KeyError if key is not present
    _ = d[key]

    index = list(d).index(key)

    keys = list(d)[0: index + 1]
    values = list(d.values())[0: index + 1]

    return dict(zip(keys, values))


def raise_if_dt_is_not_floored_to_the_first(
        dt: datetime.datetime,
        units: str,
) -> datetime.datetime:

    if dt - dt.replace(**get_dict_until(DT_REPLACE_DICT, units)):
        raise ValueError(f'datetime must be "floored" to the first {units}')
    return dt


# The functions bellow can be used as validators, e.g.:
#
# >>> @validate_types_in_func_call
# ... def func(
# ...     dt: Annotated[
# ...         datetime_like_naive_or_utc_to_naive,
# ...         AfterValidator(dt_must_be_YYYYmm01_000000),
# ...     ],
# ... ) -> None:
# ...     print(dt)

def dt_must_be_YYYYmmdd_HHMM00(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime is not 'YYYY-mm-dd HH:MM:00'."""
    return raise_if_dt_is_not_floored_to_the_first(dt, "second")


def dt_must_be_YYYYmmdd_HH0000(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime is not 'YYYY-mm-dd HH:00:00'."""
    return raise_if_dt_is_not_floored_to_the_first(dt, "minute")


def dt_must_be_YYYYmmdd_000000(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime is not 'YYYY-mm-dd 00:00:00'."""
    return raise_if_dt_is_not_floored_to_the_first(dt, "hour")


def dt_must_be_YYYYmm01_000000(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime is not 'YYYY-mm-01 00:00:00'."""
    return raise_if_dt_is_not_floored_to_the_first(dt, "day")


def dt_must_be_YYYY0101_000000(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime is not 'YYYY-01-01 00:00:00'."""
    return raise_if_dt_is_not_floored_to_the_first(dt, "month")


# -----------------------------------------------------------------------------
# @todo: create timedelta validators.
# * maybe return a dateutil.relativedelta.relativedelta object as datetime.timedelta
#   only accepts time deltas up to days.
# * the validator should accept ISO8601 str (e.g. 'P3DT12H30M5S') using pydantic
#   https://docs.pydantic.dev/latest/api/standard_library_types/#datetimetimedelta
# * maybe accept datetime.timedelta, ISO8601 str and dict (e.g.: {"months":2, "days"=3})
#   and coerce everything to dateutil.relativedelta.relativedelta
# * accept int/float as seconds

# -----------------------------------------------------------------------------
# set strict=False to allow coercion from string
path_like = Annotated[Path, Field(strict=False)]
path_like.__doc__ = (
    """Type alias to validate path_like (Path, str, etc) and coerce it to path.

    Examples
    --------
    >>> validate_type(Path("/tmp/foo/bar"), path_like)
    PosixPath('/tmp/foo/bar')

    >>> validate_type("/tmp/foo/bar", path_like)
    PosixPath('/tmp/foo/bar')

    """
)


def relative_path_to_absolute(path: Path) -> Path:
    return path.expanduser().absolute()


path_like_absolute = Annotated[path_like, AfterValidator(relative_path_to_absolute)]
path_like_absolute.__doc__ = (
    """Same as path_like, but return an absolute path.

    Examples
    --------
    >>> validate_type("foo/bar", path_like_absolute)
    PosixPath('/home/user/foo/bar')

    """
)


# @todo: create some more path types, e.g.:
# * pydantic FilePath : must exist and be a file
# * pydantic DirectoryPath: must exist and be a directory
# * pydantic NewPath : must be new and parent exist (maybe turn off parent necessity)
# * Path_must_be_overwritable
# * check https://github.com/xfrenette/pathtype for ideas
