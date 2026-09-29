"""Разбиение упорядоченного датафрейма на чанки без разрыва групп."""

from __future__ import annotations

import operator
from collections.abc import Generator, Hashable

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray

__all__ = ["iter_chunks"]


def iter_chunks(
    df: pd.DataFrame,
    by: Hashable,
    size: int,
    *,
    assume_sorted: bool = False,
) -> Generator[pd.DataFrame, None, None]:
    """Режет `df` на непрерывные чанки по колонке `by`.

    Строки с одинаковым значением ключа всегда остаются в одном чанке, поэтому
    значения не пересекаются между чанками. Каждый чанк, кроме последнего,
    содержит не меньше `size` строк; последний короче, если строк не осталось.
    Если строк не больше `size`, фрейм возвращается целиком одним чанком.

    Колонка `by` должна быть монотонной, по возрастанию или по убыванию:
    конец группы ищется по равенству значений, направление ему не важно.
    `assume_sorted` пропускает проверку порядка, если он гарантирован выше по
    стеку; пропуски в ключе тогда ищутся только с краю.

    Аргументы проверяются сразу при вызове, чанки считаются лениво. Время O(n):
    проверка порядка линейная, а граница чанка ищется за O(log k), где k это
    длина группы на границе. Дополнительной памяти O(1): чанки это срезы без
    копирования.

    :raises TypeError: `size` не целое число.
    :raises ValueError: `size` меньше 1, в ключе пропуски или он не монотонный.
    :raises KeyError: колонки `by` нет во фрейме.
    """
    size = operator.index(size)
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")

    keys = df[by]
    _check_keys(keys, by, assume_sorted)
    return _chunks(df, keys, size)


def _check_keys(keys: pd.Series, by: Hashable, assume_sorted: bool) -> None:
    if keys.empty:
        return
    missing = f"column {by!r} must not contain missing values"
    # после sort_values пропуски стоят с краю, их видно без прохода по колонке
    if pd.isna(keys.iat[0]) or pd.isna(keys.iat[-1]):
        raise ValueError(missing)
    if assume_sorted:
        return

    # Index считает оба направления за один проход и запоминает результат
    index = pd.Index(keys)
    if index.is_monotonic_increasing or index.is_monotonic_decreasing:
        return
    # пропуск в середине тоже ломает монотонность; маску строим только перед ошибкой
    if keys.hasnans:
        raise ValueError(missing)
    raise ValueError(f"column {by!r} must be sorted")


def _chunks(df: pd.DataFrame, keys: pd.Series, size: int) -> Generator[pd.DataFrame, None, None]:
    n = len(df)
    if n <= size:
        # срез, а не сам df: как и остальные чанки, это отдельный объект
        yield df.iloc[:]
        return

    # to_numpy() отдаёт view только для numpy-типов; tz, category и строки читаем через .array
    values = keys.to_numpy() if isinstance(keys.dtype, np.dtype) else keys.array
    start = 0
    while start + size < n:
        # последняя строка, обязанная попасть в чанк, и конец её группы
        stop = _group_end(values, start + size - 1)
        yield df.iloc[start:stop]
        start = stop
    if start < n:
        yield df.iloc[start:n]


def _group_end(values: np.ndarray | ExtensionArray, pos: int) -> int:
    """Позиция сразу за группой строки `pos`.

    Галопирующий поиск: шаг удваивается, пока значение совпадает с `values[pos]`,
    потом бинарный поиск на последнем отрезке. Результат всегда больше `pos`.
    """
    n = len(values)
    value = values[pos]
    lo, hi, step = pos, pos + 1, 1
    while hi < n and values[hi] == value:
        lo, step = hi, step * 2
        hi = pos + step
    hi = min(hi, n)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if values[mid] == value:
            lo = mid
        else:
            hi = mid
    return hi
