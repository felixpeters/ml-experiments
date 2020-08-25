import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames
from features import dtypes, missing_vals


@given(
    data_frames([
        column('A', dtype=int),
        column('B', dtype=float),
        column('C', dtype=int),
    ])
)
def test_dtypes(df):
    types = dtypes(df)
    assert types[0][0] == 'int64'
    assert types[0][1] == 2
    assert len(types) == 2


@given(
    data_frames([
        column(elements=st.one_of(st.none(), st.integers())),
        column(elements=st.one_of(st.none(), st.floats())),
        column(elements=st.one_of(st.none(), st.emails())),
    ])
)
def test_missing_vals(df):
    missing = missing_vals(df, axis=0)
    assert len(missing) == df.shape[1]
    if df.empty:
        assert missing.isna().all()
    else:
        assert missing.between(0, 1, inclusive=True).all()
