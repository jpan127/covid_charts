import asyncio
import csv
import dataclasses
import datetime
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Mapping, Tuple, TypeVar, Generic, Optional

import aiohttp
import requests

T = TypeVar("T")
@dataclasses.dataclass
class Statistics(Generic[T]):
    date: datetime.datetime = datetime.datetime(2020, 1, 1)
    death: T = 0
    hospitalizedCurrently: T = 0
    inIcuCurrently: T = 0
    # positive: T = 0
    positiveIncrease: T = 0

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Statistics[T]":
        d = {k: v for k, v in d.items() if k in Statistics.fields()}
        obj = Statistics[T](**d)
        if isinstance(obj.date, int):
            obj.date = datetime.datetime(2020, 1, 1)
        elif isinstance(obj.date, str):
            obj.date = datetime.datetime.strptime(obj.date.split(" ")[0], "%Y-%m-%d")
        return obj

    @staticmethod
    def fields() -> List[str]:
        return [field.name for field in dataclasses.fields(Statistics)]

    @staticmethod
    def color(field: str) -> str:
        # https://coolors.co/533a71-6184d8-50c5b7-9cec5b-e5c1bd-f4e952
        COLORS = ["#533a71", "#6184d8", "#50c5b7", "#9cec5b", "#F4E952", "#E5C1BD"]
        COLOR_MAP = {
            field : color for field, color in zip(Statistics.fields(), COLORS)
        }
        return COLOR_MAP[field]

    def as_dict(self) -> Mapping[str, str]:
        return {field : str(getattr(self, field)) for field in self.fields()}

@dataclasses.dataclass
class AggregatedStatistics:
    death: List[int] = dataclasses.field(default_factory=list)
    hospitalizedCurrently: List[int] = dataclasses.field(default_factory=list)
    inIcuCurrently: List[int] = dataclasses.field(default_factory=list)
    # positive: List[int] = dataclasses.field(default_factory=list)
    positiveIncrease: List[int] = dataclasses.field(default_factory=list)

    @staticmethod
    def fields() -> List[str]:
        return [field.name for field in dataclasses.fields(AggregatedStatistics)]

@dataclasses.dataclass
class Metrics:
    average: Statistics[float] = Statistics[float]()
    percent_delta: Statistics[float] = Statistics[float]()

    @staticmethod
    def fields() -> List[str]:
        return [field.name for field in dataclasses.fields(Metrics)]

    def as_dict(self) -> Mapping[str, str]:
        return {
            field : getattr(self, field).as_dict() for field in Metrics.fields()
        }

class Cache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache = defaultdict(dict)
        self._loads: int = 0
        self._misses: int = 0

    def __enter__(self) -> "Cache":
        if not self._path.exists():
            self._path.touch()
            return self
        with self._path.open() as f:
            try:
                self._cache.update(json.load(f))
            except json.decoder.JSONDecodeError:
                pass
        return self

    def __exit__(self, *args) -> None:
        with self._path.open('w') as f:
            json.dump(self._cache, f, indent=4)
        print(f"Cache misses : {float(self._misses) / self._loads}")

    def load(self, state: str, date: datetime.datetime) -> "Optional[Statistics[int]]":
        self._loads += 1
        try:
            return Statistics[int].from_dict(self._cache[state][str(date)])
        except KeyError:
            self._misses += 1
            return None

    def store(self, state: str, date: datetime.datetime, data: Statistics[int]) -> None:
        self._cache[state][str(date)] = data.as_dict()

def generate_metrics(aggregate: AggregatedStatistics) -> Metrics:
    metrics = Metrics(Statistics())
    for field in Statistics.fields():
        if field == "date":
            continue
        values = getattr(aggregate, field)
        values = [float(v) for v in values if v not in [None, "None"]]
        if not values:
            continue
        setattr(metrics.average, field, statistics.mean(values))
        setattr(metrics.percent_delta, field, round((values[-1] - values[0]) / values[0], 4) * 100.0)
    return metrics

def decrement_date_iterator(days: int) -> List[datetime.datetime]:
    ONE_DAY_DELTA = datetime.timedelta(1)
    today = datetime.datetime.now()
    for i in reversed(range(1, days + 1)):
        yield today - (ONE_DAY_DELTA * i)

def aggregate_data(data: List[Statistics[int]]) -> Tuple[List[datetime.datetime], AggregatedStatistics]:
    aggregate = AggregatedStatistics()
    for field in AggregatedStatistics.fields():
        setattr(aggregate, field, [getattr(stat, field) for stat in data])

    dates = [stat.date for stat in data]
    return (dates, aggregate)

async def get(type: str, state: str) -> Statistics[int]:
    async with aiohttp.ClientSession() as session:
        url = f"https://api.covidtracking.com/v1/states/{state}/{type}.json"
        async with session.get(url) as response:
            data = json.loads(await response.text())
            try:
                return Statistics[int].from_dict({ k : data[k] for k in Statistics.fields() })
            except KeyError:
                print(f"Data doesn't seem correct for {url}: {data}")
                import sys
                sys.exit(1)

async def get_by_date(cache: Cache, month: int, day: int, state: str) -> Statistics[int]:
    date = datetime.datetime(2020, month, day)

    maybe_cache_hit = cache.load(state, date)
    if maybe_cache_hit is not None:
        return maybe_cache_hit
    queried_data = await get(f"2020{month}{'0' if day < 10 else ''}{day}", state)
    queried_data.date = date
    cache.store(state, date, queried_data)
    return queried_data

async def get_historical_data(cache: Cache, days: int, state: str) -> List[Statistics[int]]:
    return await asyncio.gather(*(
        get_by_date(cache, date.month, date.day, state) for date in decrement_date_iterator(days)
    ))

async def generate(cache: Cache, days: int, state: str) -> Tuple[List[datetime.datetime], Tuple[str, AggregatedStatistics]]:
    t = time.time()
    data = await get_historical_data(cache, days, state)
    dates, aggregate = aggregate_data(data)
    compute_time = time.time() - t
    print(f"[{days}] [{state}] Compute Time: {compute_time}")
    return dates, (state, aggregate)
