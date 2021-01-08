import math
import numpy
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
from matplotlib.cbook import flatten

from covid_charts.data import Statistics, AggregatedStatistics, DatesAndAggregatedStatistics

def plot(aggregates: List[DatesAndAggregatedStatistics], image_path: Path, show_plot: bool) -> None:
    """
    Plots the results into a 2x2 layout of charts.

    Args:
        aggregates: Aggregated statistics to be each transformed into a chart
        image_path: The path to save the image of the charts to
        show_plot : Whether to show (and block the program) the chart or not
    """
    if len(aggregates) != 4:
        raise NotImplementedError("Only 4 sets of statistics is supported right now")

    # 2x2 subplots
    _, axes = plt.subplots(2, 2)
    axes = list(flatten(axes))

    for i, (dates, (state, aggregate)) in enumerate(aggregates):
        # Log scale, the state is the subplot title
        ax = axes[i]
        ax.set_yscale('log')
        ax.title.set_text(state.upper())

        # Customize a terser date formatting for the x axis
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        for field in AggregatedStatistics.fields():
            values = getattr(aggregate, field)
            # Skip empty sets of values
            values = [int(v) if v is not None and v != "None" else 0 for v in values]
            if not values:
                continue

            # Convert the datetimes to integers so they can be used by numpy
            dates_as_nums = list(map(mdates.date2num, dates))
            # Calculate a line of best fit
            polynomial_functor = numpy.poly1d(numpy.polyfit(dates_as_nums, values, 1))
            best_fit_ys = polynomial_functor(dates_as_nums)
            # Plot the dates (x) vs values (y)
            ax.plot(dates, values, marker='o', label=field, linewidth=5, color=Statistics.color(field))
            # Plot the line of best fit with the same x axis
            ax.plot(dates, best_fit_ys, linestyle=':', linewidth=10, alpha=0.5, color=Statistics.color(field))
            # Annotate each value of the set of values
            for x, y in zip(dates, values):
                ax.annotate(f"{y:,}", xy=(x, y + 1))
            # Plot the slope of the best fit line
            best_fit_slope = best_fit_ys[-1] - best_fit_ys[0]
            # Annotate the slope of the best fit line, but it needs to be offsetted in the y dimension
            # so that it will be less likely to be drawn on top of another annotation
            # Offset the y of the last value of the best fit line by a scaled factor of 10
            factor_of_10 = math.log10(int(best_fit_ys[-1])) if best_fit_ys[-1] > 0.0 else 100
            y_offset = pow(10, factor_of_10 - 1)
            ax.text(dates[-1],
                round(best_fit_ys[-1]) + y_offset,
                f"{best_fit_slope:,}",
                bbox=dict(facecolor="none", edgecolor=Statistics.color(field)))

        # Slant the x axis tick annotations so it is less cramped
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

    # Save the image if it doesn't exist already
    if not image_path.exists():
        plt.gcf().set_size_inches(25.6, 14.4)
        plt.savefig(str(image_path), dpi=1000)

    if show_plot:
        plt.show()
