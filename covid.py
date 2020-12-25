import asyncio
import datetime
from pathlib import Path
from typing import List, Optional, Type, Union

import click
import tabulate

from covid_charts import gmail, data, plot

async def async_main(
        gmail_client: gmail.Resource,
        recipients: Optional[str],
        days: int,
        states: str,
        cache_path: str,
        do_plot: bool,
        show_plot: bool) -> None:
    with data.Cache(Path(cache_path)) as cache:
        ctp_api = data.CovidTrackingProjectAPI(cache)
        aggregates = await asyncio.gather(*[ctp_api.generate(days, state) for state in states.split(" ")])
        nyt_api = data.NewYorkTimesAPI(cache)
        aggregates += [nyt_api.generate(days, "ca", "Santa Clara")]
        # print(aggregates[0])
        # print(" ")
        # print(aggregates[-1])
        # return

    texts = []
    for _, (state, aggregate) in aggregates:
        metrics = data.generate_metrics(aggregate)
        d: List[List[Union[Type, str]]] = [list(filter(lambda x: x != "date", [""] + data.Statistics.fields()))]
        for metric_type, stats in metrics.as_dict().items():
            d.append([metric_type])
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
        plot.plot(aggregates, image_name, show_plot)
    if recipients:
        gmail.send(gmail_client, recipients, text, image_name)

@click.command()
@click.option("--days", default=7, help="Number of days to look back in history")
@click.option("--states", default="ca tx ny", help="Which states to look up")
@click.option("--cache_path", default="covid_data.json", help="The path to the cache file to cache results")
@click.option("--recipients", default=None, help="Emails to send results to")
@click.option("--do_plot", is_flag=True, help="Whether to plot the charts or not")
@click.option("--show_plot", is_flag=True, help="Whether to show the charts or not")
def main(days: int, states: str, cache_path: str, recipients: Optional[str], do_plot: bool, show_plot: bool) -> None:
    if recipients and not do_plot:
        raise click.UsageError("If sending email to recipients, the data must be plotted to produce charts")
    if not do_plot and show_plot:
        raise click.UsageError("Plot must be drawn to show")
    asyncio.get_event_loop().run_until_complete(
        async_main(gmail.login(), recipients, days, states, cache_path, do_plot, show_plot)
    )

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
