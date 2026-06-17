from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .utils import with_ax


@with_ax
def plot_categorical_vs_target(
	df,
	column,
	target,
	weight_column=None,
	ax=None
):
	if weight_column is None:
		cross = pd.crosstab(df[column], df[target], normalize="index") * 100
	else:
		cross = df.pivot_table(
			index=column,
			columns=target,
			values=weight_column,
			aggfunc="sum",
			fill_value=0
		)
		cross = cross.div(cross.sum(axis=1), axis=0) * 100

	cross = cross.sort_values(by=cross.columns[0], ascending=False)

	labels = cross.index
	classes = cross.columns
	bottom = np.zeros(len(labels))

	for cls in classes:
		ax.bar(labels, cross[cls].values, bottom=bottom, label=str(cls))
		bottom += cross[cls].values

	title = f"{column} vs {target} (%)"
	if weight_column is not None:
		title += " (weighted)"
	ax.set_title(title)

	ax.set_ylabel("Weighted Percentage" if weight_column else "Percentage")
	ax.set_xticks(range(len(labels)))
	ax.set_xticklabels(labels, rotation=45, ha="right")
	
	#handles, labels = ax.get_legend_handles_labels()
	ax.legend(
		#handles=handles,
		#labels=labels,
		title=target,
		loc="upper left",
		bbox_to_anchor=(1.02, 1.0),
		ncol=1,
		frameon=True,
		columnspacing=0.8,
		handletextpad=0.4,
		borderaxespad=0
	)


@with_ax
def plot_numeric_vs_target(
	df,
	column,
	target="label",
	weight_column=None,
	ax=None
):
	classes = df[target].dropna().unique()

	data = [
		df[df[target] == c][column].dropna().values
		for c in classes
	]

	ax.boxplot(data, labels=classes, patch_artist=True)

	title = f"{column} vs {target}"
	if weight_column is not None:
		title += " (weighted)"
	ax.set_title(title)

	ax.set_xlabel(target)
	ax.set_ylabel(column)


@with_ax
def plot_top_categories_vs_target(
	df,
	column,
	target,
	weight_column=None,
	top_n=10,
	ax=None
):
	if weight_column is None:
		top_categories = df[column].value_counts().head(top_n).index
	else:
		top_categories = (
			df.groupby(column)[weight_column]
			.sum()
			.sort_values(ascending=False)
			.head(top_n)
			.index
		)

	filtered = df[df[column].isin(top_categories)]

	if weight_column is None:
		cross = pd.crosstab(filtered[column], filtered[target])
	else:
		cross = filtered.pivot_table(
			index=column,
			columns=target,
			values=weight_column,
			aggfunc="sum",
			fill_value=0
		)

	categories = cross.index
	classes = cross.columns
	bottom = np.zeros(len(categories))

	for cls in classes:
		ax.bar(categories, cross[cls].values, bottom=bottom, label=str(cls))
		bottom += cross[cls].values

	title = f"Top {top_n} {column} vs {target}"
	if weight_column is not None:
		title += " (weighted)"
	ax.set_title(title)

	ax.set_xticklabels(categories, rotation=45, ha="right")
	ax.legend(title=target)

