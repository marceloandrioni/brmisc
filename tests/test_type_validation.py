import pytest

import datetime
import pytz
import numpy as np
import pandas as pd

from pydantic import ValidationError

from brmisc.type_validation import (
    validate_types_in_func_call,
    validate_type,
    int_like,
    datetime_like,
    datetime_like_naive,
    datetime_like_aware,
    datetime_like_naive_or_utc_to_naive,
    datetime_like_naive_or_utc_to_utc,
    dt_must_be_YYYYmmdd_HHMM00,
    dt_must_be_YYYYmmdd_HH0000,
    dt_must_be_YYYYmmdd_000000,
    dt_must_be_YYYYmm01_000000,
    dt_must_be_YYYY0101_000000,
    path_like,
)


def test_int_like_with_round_float():
    assert validate_type(1.0, int_like) == 1


def test_int_like_with_non_round_float():
    with pytest.raises(ValidationError, match=r".*Assertion failed, value 1.01 is not a round number.*"):
        validate_type(1.01, int_like)


def test_datetime_like_with_datetime():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6)
    dt_in = dt_out
    assert validate_type(dt_in, datetime_like) == dt_out


def test_datetime_like_with_date():
    dt_out = datetime.datetime(2001, 2, 3)
    dt_in = dt_out.date()
    assert validate_type(dt_in, datetime_like) == dt_out


def test_datetime_like_with_np_datetime64():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6)
    dt_in = np.datetime64("2001-02-03 04:05:06")
    assert validate_type(dt_in, datetime_like) == dt_out


def test_datetime_like_with_timestamp():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6)
    dt_in = pd.Timestamp("2001-02-03 04:05:06")
    assert validate_type(dt_in, datetime_like) == dt_out


def test_datetime_like_with_str():

    Y = datetime.datetime(2001, 1, 1)
    Ym = datetime.datetime(2001, 2, 1)
    Ymd = datetime.datetime(2001, 2, 3)
    YmdH = datetime.datetime(2001, 2, 3, 4)
    YmdHM = datetime.datetime(2001, 2, 3, 4, 5)
    YmdHMS = datetime.datetime(2001, 2, 3, 4, 5, 6)
    YmdHMSz = datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=pytz.UTC)
    YmdHMSf = datetime.datetime(2001, 2, 3, 4, 5, 6, 7)
    YmdHMSfz = datetime.datetime(2001, 2, 3, 4, 5, 6, 7, tzinfo=pytz.UTC)

    dts = {
        "2001": Y,
        "200102": Ym,
        "20010203": Ymd,
        "2001020304": YmdH,
        "200102030405": YmdHM,
        "20010203040506": YmdHMS,
        "20010203040506Z": YmdHMSz,
        "20010203040506+00:00": YmdHMSz,
        "20010203040506.000007": YmdHMSf,
        "20010203040506.000007Z": YmdHMSfz,

        "2001-02": Ym,
        "2001-02-03": Ymd,
        "2001-02-03 04": YmdH,
        "2001-02-03 04:05": YmdHM,
        "2001-02-03 04:05:06": YmdHMS,
        "2001-02-03 04:05:06Z": YmdHMSz,
        "2001-02-03 04:05:06+00:00": YmdHMSz,
        "2001-02-03 04:05:06.000007": YmdHMSf,
        "2001-02-03 04:05:06.000007+00:00": YmdHMSfz,

        "2001-02-03T04": YmdH,
        "2001-02-03T04:05": YmdHM,
        "2001-02-03T04:05:06": YmdHMS,
        "2001-02-03T04:05:06Z": YmdHMSz,
        "2001-02-03T04:05:06+00:00": YmdHMSz,
        "2001-02-03T04:05:06.000007": YmdHMSf,
        "2001-02-03T04:05:06.000007+00:00": YmdHMSfz,

        "03/02/2001": Ymd,
        "03/02/2001 04": YmdH,
        "03/02/2001 04:05": YmdHM,
        "03/02/2001 04:05:06": YmdHMS,
        "03/02/2001 04:05:06Z": YmdHMSz,
        "03/02/2001 04:05:06+00:00": YmdHMSz,
        "03/02/2001 04:05:06.000007": YmdHMSf,
        "03/02/2001 04:05:06.000007+00:00": YmdHMSfz,
    }

    for dt_in, dt_out in dts.items():
        assert validate_type(dt_in, datetime_like) == dt_out


def test_datetime_like_naive_with_naive():
    assert validate_type("2001-02-03 04:05:06", datetime_like_naive)


def test_datetime_like_naive_with_aware():
    with pytest.raises(ValidationError, match=r".*Input should not have timezone info.*"):
        validate_type("2001-02-03 04:05:06Z", datetime_like_naive)


def test_datetime_like_aware_with_aware():
    assert validate_type("2001-02-03 04:05:06Z", datetime_like_aware)


def test_datetime_like_aware_with_naive():
    with pytest.raises(ValidationError, match=r".*Input should have timezone info.*"):
        validate_type("2001-02-03 04:05:06", datetime_like_aware)


def test_datetime_like_with_number():
    with pytest.raises(ValidationError, match=r".*Input should be a valid datetime.*"):
        validate_type(1.0, datetime_like)


def test_datetime_like_naive_or_utc_to_naive_with_naive():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6)
    dt_in = validate_type("2001-02-03 04:05:06", datetime_like_naive_or_utc_to_naive)
    assert dt_in == dt_out


def test_datetime_like_naive_or_utc_to_naive_with_utc():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6)
    dt_in = validate_type("2001-02-03 04:05:06Z", datetime_like_naive_or_utc_to_naive)
    assert dt_in == dt_out


def test_datetime_like_naive_or_utc_to_naive_with_non_utc():
    with pytest.raises(ValidationError, match=r".*Input should have UTC timezone.*"):
        validate_type("2001-02-03 04:05:06-03:00", datetime_like_naive_or_utc_to_naive)


def test_datetime_like_naive_or_utc_to_utc_with_naive():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)
    dt_in = validate_type("2001-02-03 04:05:06", datetime_like_naive_or_utc_to_utc)
    assert dt_in == dt_out


def test_datetime_like_naive_or_utc_to_utc_with_utc():
    dt_out = datetime.datetime(2001, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)
    dt_in = validate_type("2001-02-03 04:05:06Z", datetime_like_naive_or_utc_to_utc)
    assert dt_in == dt_out


def test_datetime_like_naive_or_utc_to_utc_with_non_utc():
    with pytest.raises(ValidationError, match=r".*Input should have UTC timezone.*"):
        validate_type("2001-02-03 04:05:06-03:00", datetime_like_naive_or_utc_to_naive)
