"""Разбиение упорядоченного датафрейма на чанки без разрыва групп."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray

__all__ = ["iter_chunks"]


def iter_chunks(
    df: pd.DataFrame,
    by: str,
    size: int,
    *,
    assume_sorted: bool = False,
) -> Generator[pd.DataFrame, None, None]:
    """Режет `df` на непрерывные куски по колонке `by`.

    Строки с одинаковым значением ключа всегда остаются в одном куске, поэтому
    значения не пересекаются между чанками. Каждый чанк, кроме последнего,
    содержит не меньше `size` строк; последний короче, если строк не осталось.
    Если строк не больше `size`, фрейм возвращается целиком одним чанком.

    Колонка `by` должна быть монотонной, по возрастанию или по убыванию:
    конец группы ищется по равенству значений, направление ему не важно.
    `assume_sorted` пропускает проверку, если порядок гарантирован выше по стеку.

    Время O(n) на проверки входа плюс O(log k) на чанк, где k это длина группы
    на границе. Список чанков не строится, чанки это срезы без копирования.

    :raises ValueError: `size` меньше 1, в ключе пропуски или он не монотонный.
    :raises KeyError: колонки `by` нет во фрейме.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")

    keys = df[by]
    n = len(df)
    if n <= size:
        yield df
        return

    if keys.hasnans:
        raise ValueError(f"column {by!r} must not contain missing values")
    if not assume_sorted and not (keys.is_monotonic_increasing or keys.is_monotonic_decreasing):
        raise ValueError(f"column {by!r} must be sorted")

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
