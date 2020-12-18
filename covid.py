import asyncio
import datetime
from pathlib import Path
from typing import Optional

import click
import tabulate

from covid_charts import gmail, data, plot

async def async_main(
    gmail_client: gmail.Client,
    recipients: Optional[str],
    days: int,
    states: str,
    cache_path: str,
    do_plot: bool) -> None:
    with data.Cache(Path(cache_path)) as cache:
        aggregates = await asyncio.gather(*[data.generate(cache, days, state) for state in states.split(" ")])

    texts = []
    for _, (state, aggregate) in aggregates:
        metrics = data.generate_metrics(aggregate)
        d = [filter(lambda x: x != "date", [""] + data.Statistics.fields())]
        for type, stats in metrics.as_dict().items():
            d.append([type])
            for field in data.Statistics.fields():
                if field != "date":
                    d[-1].append(stats[field])

        texts.append(
            f"{days}-day Analysis ({state}):\n{tabulate.tabulate(d, headers='firstrow', tablefmt='fancy_grid')}\n")

    text = "\n".join(texts)
    print(text)
    date = datetime.datetime.now()
    image_name = Path(f"{date.month}-{date.day}_{days}_{states.replace(' ', '-')}.jpg")
    if do_plot:
        plot.plot(aggregates, image_name)
    if recipients:
        gmail.send(gmail_client, recipients, text, image_name)

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
        async_main(gmail.login(), recipients, days, states, cache_path, do_plot)
    )

if __name__ == "__main__":
    main()
