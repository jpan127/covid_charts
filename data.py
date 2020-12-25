import asyncio
import csv
import dataclasses
import datetime
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import overload, Any, Union, Dict, List, Generator, Mapping, Tuple, Type, TypeVar, Generic, Optional

import aiohttp
import requests

T = TypeVar("T", float, int)
# @TODO: Change this into a dataclass
DatesAndAggregatedStatistics = Tuple[List[datetime.datetime], Tuple[str, "AggregatedStatistics"]]

@dataclasses.dataclass
class Statistics(Generic[T]):
    Ts = Union["Statistics[int]", "Statistics[float]"]

    death: T
    hospitalizedCurrently: T
    inIcuCurrently: T
    # positive: T
    positiveIncrease: T
    date: datetime.datetime = datetime.datetime(2020, 1, 1)

    @staticmethod
    @overload
    def make(t: Type[int]) -> "Statistics[int]": ...
    @staticmethod
    @overload
    def make(t: Type[float]) -> "Statistics[float]": ...
    @staticmethod
    def make(t: Union[Type[int], Type[float]]) -> "Statistics.Ts":
        if isinstance(t, type(int)):
            return Statistics[int](0, 0, 0, 0)
        if isinstance(t, type(float)):
            return Statistics[float](0.0, 0.0, 0.0, 0.0)
        raise TypeError("")

    @staticmethod
    @overload
    def from_dict(t: Type[int], d: Mapping[str, Any]) -> "Statistics[int]": ...
    @staticmethod
    @overload
    def from_dict(t: Type[float], d: Mapping[str, Any]) -> "Statistics[float]": ...
    @staticmethod
    def from_dict(t: Union[Type[int], Type[float]], d: Mapping[str, Any]) -> "Statistics.Ts":
        d = {k: v for k, v in d.items() if k in Statistics.fields()}
        for field in Statistics.fields():
            if field not in d:
                d[field] = 0
        obj: Optional[Statistics.Ts] = None
        if isinstance(t, type(int)):
            obj = Statistics[int](**d)
        elif isinstance(t, type(float)):
            obj = Statistics[float](**d)
        else:
            raise TypeError("")

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
        COLOR_MAP = dict(zip(Statistics.fields(), COLORS))
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
    average: Statistics[float] = dataclasses.field(default_factory=lambda: Statistics.make(float))
    percent_delta: Statistics[float] = dataclasses.field(default_factory=lambda: Statistics.make(float))

    @staticmethod
    def fields() -> List[str]:
        return [field.name for field in dataclasses.fields(Metrics)]

    def as_dict(self) -> Mapping[str, Any]:
        return {
            field : getattr(self, field).as_dict() for field in Metrics.fields()
        }

class State:
    STATE_MAP = {
        "AL" : "Alabama",
        "AK" : "Alaska",
        "AZ" : "Arizona",
        "AR" : "Arkansas",
        "CA" : "California",
        "CO" : "Colorado",
        "CT" : "Connecticut",
        "DE" : "Delaware",
        "FL" : "Florida",
        "GA" : "Georgia",
        "HI" : "Hawaii",
        "ID" : "Idaho",
        "IL" : "Illinois",
        "IN" : "Indiana",
        "IA" : "Iowa",
        "KS" : "Kansas",
        "KY" : "Kentucky",
        "LA" : "Louisiana",
        "ME" : "Maine",
        "MD" : "Maryland",
        "MA" : "Massachusetts",
        "MI" : "Michigan",
        "MN" : "Minnesota",
        "MS" : "Mississippi",
        "MO" : "Missouri",
        "MT" : "Montana",
        "NE" : "Nebraska",
        "NV" : "Nevada",
        "NH" : "New Hampshire",
        "NJ" : "New Jersey",
        "NM" : "New Mexico",
        "NY" : "New York",
        "NC" : "North Carolina",
        "ND" : "North Dakota",
        "OH" : "Ohio",
        "OK" : "Oklahoma",
        "OR" : "Oregon",
        "PA" : "Pennsylvania",
        "RI" : "Rhode Island",
        "SC" : "South Carolina",
        "SD" : "South Dakota",
        "TN" : "Tennessee",
        "TX" : "Texas",
        "UT" : "Utah",
        "VT" : "Vermont",
        "VA" : "Virginia",
        "WA" : "Washington",
        "WV" : "West Virginia",
        "WI" : "Wisconsin",
        "WY" : "Wyoming",
    }

    def __init__(self, state: str) -> None:
        if state.upper() not in State.STATE_MAP:
            raise ValueError(f"{state} is not a valid state abbreviation")
        self._state = state
    def __str__(self) -> str:
        return self._state

class Cache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: defaultdict = defaultdict(dict)
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

    def _fetch(self, state: State, construct: bool, county: Optional[str] = None) -> Dict[str, Any]:
        state_data = self._cache[str(state)]
        if county:
            if construct and str(county) not in state_data:
                state_data[str(county)] = defaultdict(dict)
            return state_data[str(county)]
        return state_data

    def load(self, state: State, date: datetime.datetime, county: Optional[str] = None) -> "Optional[Statistics[int]]":
        self._loads += 1
        try:
            return Statistics.from_dict(int, self._fetch(state, False, county)[str(date)])
        except KeyError:
            print("missed", date, county)
            self._misses += 1
            return None

    def store(self, state: State, date: datetime.datetime, data: Statistics[int], county: Optional[str] = None) -> None:
        self._fetch(state, True, county)[str(date)] = data.as_dict()

def generate_metrics(aggregate: AggregatedStatistics) -> Metrics:
    metrics = Metrics()
    for field in Statistics.fields():
        if field == "date":
            continue
        values = getattr(aggregate, field)
        values = [float(v) for v in values if v not in [None, "None"]]
        if not values or all(v == 0 for v in values):
            continue
        setattr(metrics.average, field, statistics.mean(values))
        setattr(metrics.percent_delta, field, round((values[-1] - values[0]) / values[0], 4) * 100.0)
    return metrics

def reverse_date_iterator(days: int) -> Generator[datetime.datetime, None, None]:
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

class CovidTrackingProjectAPI:
    def __init__(self, cache: Cache) -> None:
        self.cache = cache

    async def _get(self, query_type: str, state: str) -> Statistics[int]:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.covidtracking.com/v1/states/{state}/{query_type}.json"
            async with session.get(url) as response:
                data = json.loads(await response.text())
                try:
                    return Statistics.from_dict(int, { k : data[k] for k in Statistics.fields() })
                except KeyError:
                    print(f"Data doesn't seem correct for {url}: {data}")
                    sys.exit(1)

    async def _get_by_date(self, month: int, day: int, state: str) -> Statistics[int]:
        date = datetime.datetime(2020, month, day)

        maybe_cache_hit = self.cache.load(State(state), date)
        if maybe_cache_hit is not None:
            return maybe_cache_hit
        queried_data = await self._get(f"2020{month}{'0' if day < 10 else ''}{day}", state)
        queried_data.date = date
        self.cache.store(State(state), date, queried_data)
        return queried_data

    async def _get_historical_data(self, days: int, state: str) -> List[Statistics[int]]:
        return await asyncio.gather(*(
            self._get_by_date(date.month, date.day, state) for date in reverse_date_iterator(days)
        ))

    async def generate(self, days: int, state: str) -> DatesAndAggregatedStatistics:
        t = time.time()
        data = await self._get_historical_data(days, state)
        dates, aggregate = aggregate_data(data)
        compute_time = time.time() - t
        print(f"[{days}] [{state}] Compute Time: {compute_time}")
        return dates, (state, aggregate)

class NewYorkTimesAPI:
    def __init__(self, cache: Cache) -> None:
        self.cache = cache

    def _get_by_date(
            self,
            url: str,
            key_filter: str,
            value: str,
            date: datetime.datetime,
            state: str,
            county: Optional[str] = None) -> Statistics[int]:
        maybe_cache_hit = self.cache.load(State(state), date, county)
        if maybe_cache_hit is not None:
            return maybe_cache_hit

        response = requests.get(url, timeout=5.0)
        response.raise_for_status()

        reader = csv.DictReader(response.text.splitlines())
        data: List[Dict[str, str]] = [row for row in reader if row[key_filter] == value]
        if not data:
            raise ValueError(f"{url} did not have results for {key_filter} == {value}")
        date_and_cases: List[Tuple[str, str]] = sorted([(d["date"], d["cases"]) for d in data])
        for prior, current in zip(date_and_cases, date_and_cases[1:]):
            current_date = datetime.datetime.strptime(current[0], "%Y-%m-%d")
            self.cache.store(
                State(state),
                current_date,
                Statistics.from_dict(int, {"date": current_date, "positiveIncrease" : int(current[1]) - int(prior[1])}),
                county
            )

        maybe_cache_hit = self.cache.load(State(state), date, county)
        assert maybe_cache_hit is not None
        return maybe_cache_hit

    def county(self, month: int, day: int, state: str, county: str) -> Statistics[int]:
        date = datetime.datetime(2020, month, day)
        data = self._get_by_date(
            "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv",
            "county",
            county,
            date,
            state,
            county
        )
        print(data)
        return data

    # def state(self, month: int, day: int, state: str) -> Statistics[int]:
    #     date = datetime.datetime(2020, month, day)
    #     data = self._get_by_date(
    #         url="https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv",
    #         key_filter="state",
    #         value=state,
    #         date=date,
    #         state=state
    #     )
    #     print(data)
    #     return data

    def _get_historical_data(self, days: int, state: str, county: str) -> List[Statistics[int]]:
        return [
            self.county(date.month, date.day, state, county) for date in reverse_date_iterator(days)
        ]

    def generate(self, days: int, state: str, county: str) -> DatesAndAggregatedStatistics:
        t = time.time()
        data = self._get_historical_data(days, state, county)
        dates, aggregate = aggregate_data(data)
        compute_time = time.time() - t
        print(f"[{days}] [{state}] Compute Time: {compute_time}")
        return dates, (county, aggregate)
