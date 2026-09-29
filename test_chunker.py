import tracemalloc
from itertools import islice

import numpy as np
import pandas as pd
import pytest

import chunker
from chunker import iter_chunks


def frame(repeats, start="2023-01-01 00:00:01", descending=False):
    ts = pd.date_range(start, periods=len(repeats), freq="s")
    df = pd.DataFrame({"dt": ts.repeat(repeats), "v": range(sum(repeats))})
    if descending:
        df = df.iloc[::-1]
    return df


def chunk_sizes(df, by, size, **kw):
    return [len(c) for c in iter_chunks(df, by, size, **kw)]


@pytest.fixture
def example():
    """Фрейм из условия: 00:00:01 x2, 00:00:02 x3, 00:00:03 x1."""
    return frame([2, 3, 1])


# кейсы из условия

@pytest.mark.parametrize(
    "size, expected",
    [
        (1, [2, 3, 1]),
        (2, [2, 3, 1]),
        (3, [5, 1]),
        (4, [5, 1]),
        (5, [5, 1]),
        (6, [6]),
        (7, [6]),
        (100, [6]),
    ],
)
def test_example_from_task(example, size, expected):
    assert chunk_sizes(example, "dt", size) == expected


def test_example_chunk_contents(example):
    first, second, third = iter_chunks(example, "dt", 1)
    assert first["v"].tolist() == [0, 1]
    assert second["v"].tolist() == [2, 3, 4]
    assert third["v"].tolist() == [5]


# инварианты

@pytest.mark.parametrize("descending", [False, True], ids=["asc", "desc"])
@pytest.mark.parametrize("repeats", [[1] * 6, [6], [5, 1], [1, 5], [2, 2, 2], [1, 4, 1], [1, 9, 2]])
@pytest.mark.parametrize("size", [1, 2, 3, 4, 7])
def test_invariants(repeats, size, descending):
    df = frame(repeats, descending=descending)
    chunks = list(iter_chunks(df, "dt", size))

    assert chunks, "генератор не должен возвращать пустую последовательность"
    assert all(len(c) for c in chunks), "пустых чанков быть не должно"
    pd.testing.assert_frame_equal(pd.concat(chunks), df)

    assert all(len(c) >= size for c in chunks[:-1])
    # чанк закрывается на первой возможной границе: без последней группы он короче size
    assert all(len(c) - (c["dt"] == c["dt"].iat[-1]).sum() < size for c in chunks)

    # каждая дата ровно в одном чанке
    assert sum(c["dt"].nunique() for c in chunks) == df["dt"].nunique()


# краевые случаи

def test_empty_frame_is_returned_as_one_chunk():
    df = pd.DataFrame({"dt": pd.to_datetime([])})
    chunks = list(iter_chunks(df, "dt", 5))
    assert len(chunks) == 1 and chunks[0].empty


# pandas 2 без Copy-on-Write предупреждает о записи в срез, это ожидаемо
@pytest.mark.filterwarnings(r"ignore:\s*A value is trying to be set on a copy of a slice")
@pytest.mark.parametrize("size", [1, 10])
def test_new_column_in_chunk_does_not_leak_into_input(example, size):
    for chunk in iter_chunks(example, "dt", size):
        chunk["x"] = 1
    assert "x" not in example


def test_single_row():
    assert chunk_sizes(frame([1]), "dt", 10) == [1]


def test_one_value_cannot_be_split():
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-01"] * 100)})
    assert chunk_sizes(df, "dt", 1) == [100]


def test_first_group_smaller_than_size_absorbs_the_next():
    assert chunk_sizes(frame([1, 5]), "dt", 2) == [6]


def test_descending_key():
    # 00:00:03 x1, 00:00:02 x3, 00:00:01 x2
    assert chunk_sizes(frame([2, 3, 1], descending=True), "dt", 3) == [4, 2]


# ошибки поднимаются сразу при вызове, без итерации

@pytest.mark.parametrize("bad, error", [(0, ValueError), (-1, ValueError), (2.5, TypeError)])
def test_bad_size_rejected(example, bad, error):
    with pytest.raises(error):
        iter_chunks(example, "dt", bad)


def test_unsorted_frame_rejected():
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-02", "2023-01-01", "2023-01-03"])})
    with pytest.raises(ValueError, match="sorted"):
        iter_chunks(df, "dt", 10)


@pytest.mark.parametrize(
    "dates, assume_sorted",
    [
        (["2023-01-01", "2023-01-02", None], False),
        ([None, "2023-01-01", "2023-01-02"], False),
        (["2023-01-01", None, "2023-01-02"], False),
        (["2023-01-01", "2023-01-02", None], True),
        ([None, "2023-01-01", "2023-01-02"], True),
    ],
)
def test_missing_values_rejected(dates, assume_sorted):
    df = pd.DataFrame({"dt": pd.to_datetime(dates)})
    with pytest.raises(ValueError, match="missing values"):
        iter_chunks(df, "dt", 2, assume_sorted=assume_sorted)


def test_assume_sorted_does_not_change_result_on_sorted_input():
    df = frame([3, 3, 3, 3])
    for size in range(1, 14):
        assert chunk_sizes(df, "dt", size, assume_sorted=True) == chunk_sizes(df, "dt", size)


def test_assume_sorted_on_unsorted_input_terminates():
    """Порядок на совести вызывающего, но зависать и терять строки нельзя."""
    df = pd.DataFrame({"k": [0, 0, 1, 0, 0]})
    chunks = list(islice(iter_chunks(df, "k", 2, assume_sorted=True), len(df) + 1))
    pd.testing.assert_frame_equal(pd.concat(chunks), df)


# ключ не обязан быть датой

@pytest.mark.parametrize(
    "keys",
    [
        [1, 1, 2, 2, 2, 3],
        ["a", "a", "b", "b", "b", "c"],
        pd.to_datetime(["2023-01-01 00:00:01"] * 2 + ["2023-01-01 00:00:02"] * 3 + ["2023-01-01 00:00:03"]).tz_localize("UTC"),
    ],
)
def test_key_dtypes(keys):
    df = pd.DataFrame({"k": keys})
    assert chunk_sizes(df, "k", 3) == [5, 1]


# память

def test_chunks_do_not_copy_data():
    df = frame([3] * 6)
    for chunk in iter_chunks(df, "dt", 6):
        assert np.shares_memory(chunk["v"].to_numpy(), df["v"].to_numpy())


@pytest.mark.parametrize("tz", [None, "UTC"])
def test_memory_does_not_grow_with_rows(tz):
    n = 2_000_000
    df = pd.DataFrame({"dt": pd.date_range("2023-01-01", periods=n // 4, freq="s", tz=tz).repeat(4)})
    tracemalloc.start()
    try:
        for _ in iter_chunks(df, "dt", 1000):
            pass
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # даже маска на байт на строку дала бы n байт
    assert peak < n // 2


def test_boundaries_are_found_lazily(monkeypatch):
    calls = []
    group_end = chunker._group_end

    def counted(*args):
        calls.append(args)
        return group_end(*args)

    monkeypatch.setattr(chunker, "_group_end", counted)
    chunks = iter_chunks(frame([10] * 1000), "dt", 10)
    next(chunks)
    assert len(calls) == 1
