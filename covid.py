import asyncio
import colorama
import dataclasses
import termcolor
import datetime
import json
import statistics
import time
import math
import numpy
import pandas
import base64
import mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Any, List, Mapping, Tuple, TypeVar, Generic, Optional

import pickle
import os.path
import googleapiclient
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import click
import aiohttp
import tabulate
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
from matplotlib.cbook import flatten

colorama.init(convert=True)

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

def plot(aggregates: Mapping[Any, Any], image_path: Path) -> None:
    _, axes = plt.subplots(2, 2)
    axes = list(flatten(axes))
    for i, (dates, (state, aggregate)) in enumerate(aggregates):
        ax = axes[i]
        ax.set_yscale('log')
        ax.title.set_text(state.upper())

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        for field in AggregatedStatistics.fields():
            values = getattr(aggregate, field)
            values = [int(v) if v is not None and v != "None" else 0 for v in values]
            if not values:
                continue

            dates_as_nums = list(map(mdates.date2num, dates))
            polynomial_functor = numpy.poly1d(numpy.polyfit(dates_as_nums, values, 1))
            best_fit_ys = polynomial_functor(dates_as_nums)
            ax.plot(dates, values, marker='o', label=field, linewidth=5, color=Statistics.color(field))
            ax.plot(dates, best_fit_ys, linestyle=':', linewidth=10, alpha=0.5, color=Statistics.color(field))
            for x, y in zip(dates, values):
                ax.annotate(f"{y:,}", xy=(x, y + 1))
            # Plot the slope of the best fit line
            best_fit_slope = best_fit_ys[-1] - best_fit_ys[0]
            factor_of_10 = math.log10(int(best_fit_ys[-1])) if best_fit_ys[-1] > 0.0 else 100
            y_offset = pow(10, factor_of_10 - 1)
            ax.text(dates[-1],
                round(best_fit_ys[-1]) + y_offset,
                f"{best_fit_slope:,}",
                bbox=dict(facecolor="none", edgecolor=Statistics.color(field)))

        for label in ax.get_xticklabels():
            label.set_rotation(40)
            label.set_horizontalalignment('right')

    # Create a deduplicated legend
    labels_handles = {
        label: handle for ax in axes for handle, label in zip(*ax.get_legend_handles_labels())
    }
    plt.figlegend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center")

    if not image_path.exists():
        plt.gcf().set_size_inches(25.6, 14.4)
        plt.savefig(str(image_path), dpi=1000)
    plt.show()

async def async_main(
    gmail: googleapiclient.discovery.Resource,
    recipients: Optional[str],
    days: int,
    states: str,
    cache_path: str,
    do_plot: bool) -> None:
    with Cache(Path(cache_path)) as cache:
        aggregates = await asyncio.gather(*[generate(cache, days, state) for state in states.split(" ")])

    texts = []
    for _, (state, aggregate) in aggregates:
        metrics = generate_metrics(aggregate)
        d = [filter(lambda x: x != "date", [""] + Statistics.fields())]
        for type, stats in metrics.as_dict().items():
            d.append([type])
            for field in Statistics.fields():
                if field != "date":
                    d[-1].append(stats[field])

        texts.append(f"{days}-day Analysis ({state}):\n{tabulate.tabulate(d, headers='firstrow', tablefmt='fancy_grid')}\n")

    text = "\n".join(texts)
    print(text)
    date = datetime.datetime.now()
    image_name = Path(f"{date.month}-{date.day}_{days}_{states.replace(' ', '-')}.jpg")
    if do_plot:
        plot(aggregates, image_name)
    if recipients:
        email(gmail, recipients, text, image_name)

def gmail_login():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', ['https://www.googleapis.com/auth/gmail.compose'])
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def email(gmail: googleapiclient.discovery.Resource, recipients: str, text: str, image_path: Path) -> None:
    html_text = "<font face='Courier New, Courier, monospace'><pre>" + text + "</pre></font>"
    html_text = html_text.replace("\n", "<br>")
    message = MIMEMultipart()
    message["to"] = recipients
    message["from"] = "jpan127@gmail.com"
    message["subject"] = f"{str(datetime.datetime.now())} Covid Metrics"
    message.attach(MIMEText(html_text, "html"))

    content_type, _ = mimetypes.guess_type(str(image_path))
    if not content_type or content_type != "image/jpeg":
        raise RuntimeError(f"{image_path} is the wrong file type")
    with image_path.open("rb") as f:
        msg = MIMEImage(f.read(), _sub_type = "jpeg")
        msg.add_header("Content-Disposition", "attachment", filename=str(image_path))
    message.attach(msg)

    message = {
        "raw": base64.urlsafe_b64encode(message.as_string().encode("utf-8")).decode("utf-8")
    }

    gmail.users().messages().send(userId="jpan127@gmail.com", body=message).execute()

@click.command()
@click.option("--days", default=7, help="Number of days to look back in history")
@click.option("--states", default="ca tx fl ny", help="Which states to look up")
@click.option("--cache_path", default="covid_data.json", help="The path to the cache file to cache results")
@click.option("--recipients", default=None, help="Emails to send results to")
@click.option("--do_plot", is_flag=True, default=None, help="Whether to plot the charts or not")
def main(days: int, states: str, cache_path: str, recipients: Optional[str], do_plot: bool) -> None:
    if recipients and not do_plot:
        raise RuntimeError("If sending email to recipients, the data must be plotted to produce charts")
    asyncio.get_event_loop().run_until_complete(
        async_main(gmail_login(), recipients, days, states, cache_path, do_plot)
    )

if __name__ == "__main__":
    main()
