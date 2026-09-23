import numpy as np
import pandas as pd
import pytest

from chunker import iter_chunks


def frame(repeats, start="2023-01-01 00:00:01"):
    ts = pd.date_range(start, periods=len(repeats), freq="s")
    return pd.DataFrame({"dt": ts.repeat(repeats), "v": range(sum(repeats))})


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

@pytest.mark.parametrize("repeats", [[1] * 6, [6], [5, 1], [1, 5], [2, 2, 2], [1, 4, 1]])
@pytest.mark.parametrize("size", [1, 2, 3, 4, 7])
def test_invariants(repeats, size):
    df = frame(repeats)
    chunks = list(iter_chunks(df, "dt", size))

    assert chunks, "генератор не должен возвращать пустую последовательность"
    assert all(len(c) for c in chunks), "пустых чанков быть не должно"
    pd.testing.assert_frame_equal(pd.concat(chunks), df)

    for chunk in chunks[:-1]:
        assert len(chunk) >= size

    # ни одна дата не встречается в двух чанках
    last_of_chunk = [c["dt"].iat[-1] for c in chunks[:-1]]
    first_of_next = [c["dt"].iat[0] for c in chunks[1:]]
    assert all(a < b for a, b in zip(last_of_chunk, first_of_next))


# краевые случаи

def test_empty_frame_yields_nothing():
    df = pd.DataFrame({"dt": pd.to_datetime([])})
    assert list(iter_chunks(df, "dt", 5)) == []


def test_single_row():
    assert chunk_sizes(frame([1]), "dt", 10) == [1]


def test_one_value_cannot_be_split():
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-01"] * 100)})
    assert chunk_sizes(df, "dt", 1) == [100]


def test_first_group_smaller_than_size_absorbs_the_next():
    assert chunk_sizes(frame([1, 5]), "dt", 2) == [6]


@pytest.mark.parametrize("bad", [0, -1])
def test_size_below_one_rejected(example, bad):
    with pytest.raises(ValueError, match="size must be >= 1"):
        list(iter_chunks(example, "dt", bad))


def test_missing_column_rejected(example):
    with pytest.raises(KeyError):
        list(iter_chunks(example, "nope", 2))


def test_unsorted_frame_rejected():
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-02", "2023-01-01"])})
    with pytest.raises(ValueError, match="sorted"):
        list(iter_chunks(df, "dt", 1))


def test_missing_values_rejected():
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-01", None])})
    with pytest.raises(ValueError, match="missing values"):
        list(iter_chunks(df, "dt", 1))


def test_assume_sorted_does_not_change_result_on_sorted_input():
    """Флаг только выключает проверку.

    На неотсортированном входе с этим флагом результат не определён: границы
    ищутся бинарным поиском. Следить за сортировкой должен вызывающий.
    """
    df = frame([3, 3, 3, 3])
    for size in range(1, 14):
        assert chunk_sizes(df, "dt", size, assume_sorted=True) == chunk_sizes(df, "dt", size)


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
    chunk = next(iter_chunks(df, "dt", 4))
    assert np.shares_memory(chunk["v"].to_numpy(), df["v"].to_numpy())


def test_generator_is_lazy():
    df = frame([10] * 1000)
    chunks = iter_chunks(df, "dt", 10)
    assert len(next(chunks)) == 10
    assert sum(len(c) for c in chunks) == len(df) - 10
