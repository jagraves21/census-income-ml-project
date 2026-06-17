from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .bar import bar_plot
from .utils import with_ax


@with_ax
def plot_categorical_distribution(
	df,
	column,
	weight_column=None,
	top_n=None,
	horizontal=False,
	sort_by_category=False,
	ax=None,
):
	if weight_column is None:
		value_counts = (
			df[column]
			.value_counts()
			.rename("count")
		)
	else:
		value_counts = (
			df.groupby(column)[weight_column]
			.sum()
			.rename("count")
		)

	if top_n is not None:
		value_counts = value_counts.head(top_n)

	if sort_by_category:
		value_counts = value_counts.sort_index()
	else:
		value_counts = value_counts.sort_values(ascending=False)

	if not pd.api.types.is_string_dtype(value_counts.index):
		value_counts.index = value_counts.index.astype(str)

	title = f"Distribution of {column}"
	if weight_column is not None:
		title += " (weighted)"

	if horizontal:
		xlabel = "Count" if weight_column is None else "Weighted Count"
		ylabel = column
	else:
		xlabel = column
		ylabel = "Count" if weight_column is None else "Weighted Count"

	bar_plot(
		df=value_counts.to_frame(),
		data_column="count",
		title=title,
		xlabel=xlabel,
		ylabel=ylabel,
		colors=None,
		ax=ax,
		horizontal=horizontal,
	)

	total = value_counts.sum()

	if horizontal:
		ax2 = ax.twiny()

		xmin, xmax = ax.get_xlim()
		ax2.set_xlim(
			xmin / total * 100,
			xmax / total * 100,
		)
		ax2.set_xlabel("Weighted Percentage" if weight_column else "Percentage")
		ax2.xaxis.set_major_formatter(PercentFormatter())
	else:
		ax2 = ax.twinx()

		ymin, ymax = ax.get_ylim()
		ax2.set_ylim(
			ymin / total * 100,
			ymax / total * 100,
		)
		ax2.set_ylabel("Weighted Percentage" if weight_column else "Percentage")
		ax2.yaxis.set_major_formatter(PercentFormatter())

	ax.text(
		0.98,
		0.98,
		f"Missing: {df[column].isna().sum()}",
		transform=ax.transAxes,
		ha="right",
		va="top",
		bbox=dict(
			boxstyle="round,pad=0.3",
			facecolor="white",
			alpha=0.8,
		),
	)


@with_ax
def plot_numeric_distribution(
	df,
	column,
	weight_column=None,
	bins=50,
	ax=None
):
	data = df[column].dropna()
	if weight_column is not None:
		weights = df.loc[data.index, weight_column]
	else:
		weights = None

	ax.hist(
		data.values,
		bins=bins,
		weights=weights,
		alpha=0.7,
		edgecolor="black"
	)

	title = f"Distribution of {column}"
	if weight_column is not None:
		title += " (weighted)"
	ax.set_title(title)

	ax.set_xlabel(column)
	ax.set_ylabel("Weighted Frequency" if weight_column else "Frequency")

