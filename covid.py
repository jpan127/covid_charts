import asyncio
import datetime
import sys
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
    """
    The main function, args are passed from the CLI.
    """
    # Open the cache file, and use it for all external queries
    with data.Cache(Path(cache_path)) as cache:
        # CTP is used for state specific data
        ctp_api = data.CovidTrackingProjectAPI(cache)
        aggregates = await asyncio.gather(*[ctp_api.generate(days, state) for state in states.split(" ")])
        # NYT is used for county level data
        nyt_api = data.NewYorkTimesAPI(cache)
        aggregates += [nyt_api.generate(days, "ca", "Santa Clara")]

    texts = []
    for _, (state, aggregate) in aggregates:
        # Generate metrics from the statistics
        metrics = data.generate_metrics(aggregate)
        # Start with the table header
        d: List[List[Union[Type, str]]] = [list(filter(lambda x: x != "date", [""] + data.Statistics.fields()))]
        for metric_type, stats in metrics.as_dict().items():
            # First column of each row is the type of the metric
            d.append([metric_type])
            # Other columns are the values
            for field in data.Statistics.fields():
                if field != "date":
                    d[-1].append(stats[field])

        # Add a table for each metric
        texts.append(
            f"{days}-day Analysis ({state}):\n{tabulate.tabulate(d, headers='firstrow', tablefmt='fancy_grid')}\n")

    # Print the tables
    text = "\n".join(texts)
    print(text.encode("utf-8").decode(sys.stdout.encoding))

    date = datetime.datetime.now().date()
    image_name = Path(f"{date.month}-{date.day}_{days}_{states.replace(' ', '-')}.jpg")
    # Plot the charts if requested
    if do_plot:
        plot.plot(aggregates, image_name, show_plot)
    # Email the recipients if requested
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
    """CLI."""
    if recipients and not do_plot:
        raise click.UsageError("If sending email to recipients, the data must be plotted to produce charts")
    if not do_plot and show_plot:
        raise click.UsageError("Plot must be drawn to show")
    asyncio.get_event_loop().run_until_complete(
        async_main(gmail.login(), recipients, days, states, cache_path, do_plot, show_plot)
    )

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
