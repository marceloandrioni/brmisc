from typing import Annotated
import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pytz
import tempfile
from decimal import Decimal

import pytest
from pydantic import ValidationError, AfterValidator
import numpy as np
import pandas as pd
import xarray as xr

from brmisc.type_validation import (
    validate_types_in_func_call,
    validate_type,
    int_like,
    datetime_like,
    datetime_like_naive,
    datetime_like_aware,
    datetime_like_naive_or_utc,
    datetime_like_naive_or_utc_to_naive,
    datetime_like_naive_or_utc_to_utc,
    dt_must_be_YYYYmmdd_HHMM00,
    dt_must_be_YYYYmmdd_HH0000,
    dt_must_be_YYYYmmdd_000000,
    dt_must_be_YYYYmm01_000000,
    dt_must_be_YYYY0101_000000,
    timedelta_like,
    path_like,
    file_path_like,
    directory_path_like,
)


def test_validate_types_in_func_call_with_correct_types():
    @validate_types_in_func_call
    def divide_by_2(x: int) -> float:
        return x / 2
    divide_by_2(5)


def test_validate_types_in_func_call_with_incorrect_in_type():
    @validate_types_in_func_call
    def divide_by_2(x: int) -> float:
        return x / 2
    with pytest.raises(ValidationError, match=r".*Input should be a valid integer.*"):
        divide_by_2(5.0)


def test_validate_types_in_func_call_with_incorrect_out_type():
    @validate_types_in_func_call
    def divide_by_2(x: int) -> int:
        return x / 2
    with pytest.raises(ValidationError, match=r".*Input should be a valid integer.*"):
        divide_by_2(5)


# -----------------------------------------------------------------------------
# int_like validation

@pytest.mark.parametrize(
    "x,y",
    [
        (1, 1),
        (1.0, 1),
        (Decimal("1.0"), 1),
        (np.int32(1), 1),
        (np.float32(1), 1),
        ([1], 1),
        ((1,), 1),
        (np.array(1), 1),
        (np.array([1]), 1),
        (np.array([[1]]), 1),
        (pd.Series(data=[1]), 1),
        (xr.DataArray(data=[1]), 1),
    ]
)
def test_int_like(x, y):
    assert validate_type(x, int_like) == y


def test_int_like_with_non_round_float():
    with pytest.raises(ValidationError, match=r".*Assertion failed, value 1.01 is not a round number.*"):
        validate_type(1.01, int_like)


# -----------------------------------------------------------------------------
# datetime_like validation

Y = datetime.datetime(2001, 1, 1)
Ym = datetime.datetime(2001, 2, 1)
Ymd = datetime.datetime(2001, 2, 3)
YmdH = datetime.datetime(2001, 2, 3, 4)
YmdHM = datetime.datetime(2001, 2, 3, 4, 5)
YmdHMS = datetime.datetime(2001, 2, 3, 4, 5, 6)
YmdHMSz = datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=pytz.UTC)
YmdHMSf = datetime.datetime(2001, 2, 3, 4, 5, 6, 7)
YmdHMSfz = datetime.datetime(2001, 2, 3, 4, 5, 6, 7, tzinfo=pytz.UTC)


@pytest.mark.parametrize(
    "dt_in,dt_out",
    [
        (datetime.datetime(2001, 2, 3, 4, 5, 6), datetime.datetime(2001, 2, 3, 4, 5, 6)),  # datetime
        (datetime.date(2001, 2, 3), datetime.datetime(2001, 2, 3)),  # date
        (np.datetime64("2001-02-03 04:05:06"), datetime.datetime(2001, 2, 3, 4, 5, 6)),  # np.datetime64
        (pd.Timestamp("2001-02-03 04:05:06"), datetime.datetime(2001, 2, 3, 4, 5, 6)),  # pd.Timestamp

        ("2001", Y),
        ("200102", Ym),
        ("20010203", Ymd),
        ("2001020304", YmdH),
        ("200102030405", YmdHM),
        ("20010203040506", YmdHMS),
        ("20010203040506Z", YmdHMSz),
        ("20010203040506+00:00", YmdHMSz),
        ("20010203040506.000007", YmdHMSf),
        ("20010203040506.000007Z", YmdHMSfz),

        ("2001-02", Ym),
        ("2001-02-03", Ymd),
        ("2001-02-03 04", YmdH),
        ("2001-02-03 04:05", YmdHM),
        ("2001-02-03 04:05:06", YmdHMS),
        ("2001-02-03 04:05:06Z", YmdHMSz),
        ("2001-02-03 04:05:06+00:00", YmdHMSz),
        ("2001-02-03 04:05:06.000007", YmdHMSf),
        ("2001-02-03 04:05:06.000007+00:00", YmdHMSfz),

        ("2001-02-03T04", YmdH),
        ("2001-02-03T04:05", YmdHM),
        ("2001-02-03T04:05:06", YmdHMS),
        ("2001-02-03T04:05:06Z", YmdHMSz),
        ("2001-02-03T04:05:06+00:00", YmdHMSz),
        ("2001-02-03T04:05:06.000007", YmdHMSf),
        ("2001-02-03T04:05:06.000007+00:00", YmdHMSfz),

        ("03/02/2001", Ymd),
        ("03/02/2001 04", YmdH),
        ("03/02/2001 04:05", YmdHM),
        ("03/02/2001 04:05:06", YmdHMS),
        ("03/02/2001 04:05:06Z", YmdHMSz),
        ("03/02/2001 04:05:06+00:00", YmdHMSz),
        ("03/02/2001 04:05:06.000007", YmdHMSf),
        ("03/02/2001 04:05:06.000007+00:00", YmdHMSfz),

    ]
)
def test_datetime_like(dt_in, dt_out):
    assert validate_type(dt_in, datetime_like) == dt_out


# -----------------------------------------------------------------------------
def test_datetime_like_naive_with_naive():
    assert validate_type("2001-02-03 04:05:06", datetime_like_naive)


def test_datetime_like_naive_with_aware():
    with pytest.raises(ValidationError, match=r".*Input should not have timezone info.*"):
        validate_type("2001-02-03 04:05:06Z", datetime_like_naive)


# -----------------------------------------------------------------------------
def test_datetime_like_aware_with_aware():
    assert validate_type("2001-02-03 04:05:06Z", datetime_like_aware)


def test_datetime_like_aware_with_naive():
    with pytest.raises(ValidationError, match=r".*Input should have timezone info.*"):
        validate_type("2001-02-03 04:05:06", datetime_like_aware)


# -----------------------------------------------------------------------------
def test_datetime_like_with_number():
    with pytest.raises(ValidationError, match=r".*Input should be a valid datetime.*"):
        validate_type(1.0, datetime_like)


# -----------------------------------------------------------------------------
def test_datetime_like_naive_or_utc_with_naive():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6)
    dt_in = validate_type("2001-02-03 04:05:06", datetime_like_naive_or_utc)
    assert dt_in == dt_out


def test_datetime_like_naive_or_utc_with_utc():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)
    dt_in = validate_type("2001-02-03 04:05:06Z", datetime_like_naive_or_utc)
    assert dt_in == dt_out


def test_datetime_like_naive_or_utc_with_non_utc():
    with pytest.raises(ValidationError, match=r".*Input should be naive or UTC.*"):
        validate_type("2001-02-03 04:05:06-03:00", datetime_like_naive_or_utc)


# -----------------------------------------------------------------------------
def test_datetime_like_naive_or_utc_to_naive_with_naive():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6)
    dt_in = validate_type("2001-02-03 04:05:06", datetime_like_naive_or_utc_to_naive)
    assert dt_in == dt_out


def test_datetime_like_naive_or_utc_to_naive_with_utc():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6)
    dt_in = validate_type("2001-02-03 04:05:06Z", datetime_like_naive_or_utc_to_naive)
    assert dt_in == dt_out


def test_datetime_like_naive_or_utc_to_naive_with_non_utc():
    with pytest.raises(ValidationError, match=r".*Input should be UTC.*"):
        validate_type("2001-02-03 04:05:06-03:00", datetime_like_naive_or_utc_to_naive)


# -----------------------------------------------------------------------------
def test_datetime_like_naive_or_utc_to_utc_with_naive():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)
    dt_in = validate_type("2001-02-03 04:05:06", datetime_like_naive_or_utc_to_utc)
    assert dt_in == dt_out


def test_datetime_like_naive_or_utc_to_utc_with_utc():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)
    dt_in = validate_type("2001-02-03 04:05:06Z", datetime_like_naive_or_utc_to_utc)
    assert dt_in == dt_out


def test_datetime_like_naive_or_utc_to_utc_with_non_utc():
    with pytest.raises(ValidationError, match=r".*Input should be UTC.*"):
        validate_type("2001-02-03 04:05:06-03:00", datetime_like_naive_or_utc_to_utc)


# -----------------------------------------------------------------------------
# rounded time validation

def test_datetime_like_YYYYmmdd_HHMM00_with_YYYYmmdd_HHMM00():
    validator = Annotated[
        datetime_like,
        AfterValidator(dt_must_be_YYYYmmdd_HHMM00),
    ]
    validate_type("2001-02-03 04:05:00", validator)


def test_datetime_like_YYYYmmdd_HHMM00_with_YYYYmmdd_HHMMSS():
    validator = Annotated[
        datetime_like,
        AfterValidator(dt_must_be_YYYYmmdd_HHMM00),
    ]
    with pytest.raises(ValidationError, match=r'.*datetime must be "floored" to the first second.*'):
        validate_type("2001-02-03 04:05:06", validator)


def test_datetime_like_YYYYmmdd_HH0000_with_YYYYmmdd_HH0000():
    validator = Annotated[
        datetime_like,
        AfterValidator(dt_must_be_YYYYmmdd_HH0000),
    ]
    validate_type("2001-02-03 04:00:00", validator)


def test_datetime_like_YYYYmmdd_HH0000_with_YYYYmmdd_HHMMSS():
    validator = Annotated[
        datetime_like,
        AfterValidator(dt_must_be_YYYYmmdd_HH0000),
    ]
    with pytest.raises(ValidationError, match=r'.*datetime must be "floored" to the first minute.*'):
        validate_type("2001-02-03 04:05:06", validator)


def test_datetime_like_YYYYmmdd_000000_with_YYYYmmdd_000000():
    validator = Annotated[
        datetime_like,
        AfterValidator(dt_must_be_YYYYmmdd_000000),
    ]
    validate_type("2001-02-03 00:00:00", validator)


def test_datetime_like_YYYYmmdd_000000_with_YYYYmmdd_HHMMSS():
    validator = Annotated[
        datetime_like,
        AfterValidator(dt_must_be_YYYYmmdd_000000),
    ]
    with pytest.raises(ValidationError, match=r'.*datetime must be "floored" to the first hour.*'):
        validate_type("2001-02-03 04:05:06", validator)


def test_datetime_like_YYYYmm01_000000_with_YYYYmm01_000000():
    validator = Annotated[
        datetime_like,
        AfterValidator(dt_must_be_YYYYmm01_000000),
    ]
    validate_type("2001-02-01 00:00:00", validator)


def test_datetime_like_YYYYmm01_000000_with_YYYYmmdd_HHMMSS():
    validator = Annotated[
        datetime_like,
        AfterValidator(dt_must_be_YYYYmm01_000000),
    ]
    with pytest.raises(ValidationError, match=r'.*datetime must be "floored" to the first day.*'):
        validate_type("2001-02-03 04:05:06", validator)


def test_datetime_like_YYYY0101_000000_with_YYYYmm01_000000():
    validator = Annotated[
        datetime_like,
        AfterValidator(dt_must_be_YYYY0101_000000),
    ]
    validate_type("2001-01-01 00:00:00", validator)


def test_datetime_like_YYYY0101_000000_with_YYYYmmdd_HHMMSS():
    validator = Annotated[
        datetime_like,
        AfterValidator(dt_must_be_YYYY0101_000000),
    ]
    with pytest.raises(ValidationError, match=r'.*datetime must be "floored" to the first month.*'):
        validate_type("2001-02-03 04:05:06", validator)


# -----------------------------------------------------------------------------
# timedelta_like validation

@pytest.mark.parametrize(
    "td_in,td_out",
    [
        (relativedelta(days=1, seconds=-1), relativedelta(days=1, seconds=-1)),
        (datetime.timedelta(days=1, seconds=-1), relativedelta(days=1, seconds=-1)),
        ("1d,02:03:04", relativedelta(days=1, hours=2, minutes=3, seconds=4)),
        ("P1DT2H3M4S", relativedelta(days=1, hours=2, minutes=3, seconds=4)),
        ({"days": 1, "seconds": -1}, relativedelta(days=1, seconds=-1)),
    ]
)
def test_timedelta_like(td_in, td_out):

    # Note: with relativedelta the same time interval can be shown in differrent
    # ways, e.g.;
    # relativedelta(hours=+23, minutes=+59, seconds=+59) != relativedelta(days=+1, seconds=-1)
    # so add it to a date to make sure it is the actually the same
    dt = datetime.datetime(2001, 1, 1)

    assert validate_type(td_in, timedelta_like) + dt == td_out + dt


# -----------------------------------------------------------------------------
# path_like validation

@pytest.mark.parametrize(
    "path_in,path_out",
    [
        (Path("/tmp/foo/bar"), Path("/tmp/foo/bar")),
        ("/tmp/foo/bar", Path("/tmp/foo/bar")),
    ]
)
def test_path_like(path_in, path_out):
    assert validate_type(path_in, path_like) == path_out


# -----------------------------------------------------------------------------
def test_file_path_like_with_file():

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write("Hello World!")
        name = temp_file.name

    assert validate_type(name, file_path_like) == Path(name)
    Path(name).unlink()


def test_file_path_like_with_missing_file():

    with tempfile.NamedTemporaryFile(mode="w", delete=True) as temp_file:
        temp_file.write("Hello World!")
        name = temp_file.name

    with pytest.raises(ValidationError, match=r'.*Path does not point to a file.*'):
        validate_type(name, file_path_like)


# -----------------------------------------------------------------------------
def test_directory_path_like_with_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        assert validate_type(temp_dir, directory_path_like) == Path(temp_dir)


def test_directory_path_like_with_missing_directory():

    with tempfile.TemporaryDirectory() as temp_dir:
        pass

    with pytest.raises(ValidationError, match=r'.*Path does not point to a directory.*'):
        validate_type(temp_dir, directory_path_like)
