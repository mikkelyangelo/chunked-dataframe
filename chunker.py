"""Разбиение отсортированного датафрейма на чанки без разрыва групп."""

from __future__ import annotations

from collections.abc import Generator

import pandas as pd

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

    Колонка `by` должна быть отсортирована по возрастанию: границы групп
    ищутся бинарным поиском. `assume_sorted` пропускает проверку, если
    сортировка гарантирована выше по стеку.

    Время O(n) на проверки входа плюс O(log n) на поиск границы каждого чанка.
    Список чанков не строится, чанки это срезы без копирования.

    :raises ValueError: `size` меньше 1, в ключе пропуски или нет сортировки.
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
    if not assume_sorted and not keys.is_monotonic_increasing:
        raise ValueError(f"column {by!r} must be sorted in ascending order")

    start = 0
    while start + size < n:
        # последняя строка, обязанная попасть в чанк, и конец её группы
        pivot = keys.iat[start + size - 1]
        stop = int(keys.searchsorted(pivot, side="right"))
        yield df.iloc[start:stop]
        start = stop
    if start < n:
        yield df.iloc[start:n]
