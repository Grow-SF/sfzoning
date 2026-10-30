"""Series -> reader / role / era table (spec: "Builder-check questions, settled", item 1)."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SeriesInfo:
    series: str
    reader: Literal["A", "B", "skip"]
    role: Literal["use", "height", "skip"]
    year: int       # research-CSV era lookup year (start year of the period; 1935 uses 1929-1946)
    period: str     # zoning and height series of one period share it


def _build() -> dict[str, SeriesInfo]:
    t: dict[str, SeriesInfo] = {}
    for y in (1921, 1929, 1935, 1948):
        t[str(y)] = SeriesInfo(str(y), "A", "use", y, str(y))
    for y in (1960, 1964, 1968, 1972, 1975, 1980):
        t[str(y)] = SeriesInfo(str(y), "B", "use", y, str(y))
    for period, year in (("1980-1990", 1980), ("1990-1996", 1990),
                         ("1996-2000", 1996), ("2000-digital", 2000)):
        t[f"{period}-zoning"] = SeriesInfo(f"{period}-zoning", "B", "use", year, period)
        t[f"{period}-height"] = SeriesInfo(f"{period}-height", "B", "height", year, period)
        t[f"{period}-su"] = SeriesInfo(f"{period}-su", "skip", "skip", year, period)
    return t


SERIES: dict[str, SeriesInfo] = _build()
